"""
MiniCPM-V-4_5 triplet inference script (counterpart of get_triplets_sft_e2e.py).

v2 uses Qwen-VL's AutoModelForImageTextToText + AutoProcessor + process_vision_info,
MiniCPM-V4.5 needs AutoModel(trust_remote_code) + AutoTokenizer + model.chat().
The rest of the CLI / output format / parsing matches v2, so upstream eval scripts can be reused.
"""
import json
import os
import re
import argparse
from lora.prompt import get_e2e_prompt
import time
from pathlib import Path
from PIL import Image
import multiprocessing as mp
from multiprocessing import Queue


_TRIPLET_RE = re.compile(r'<\s*([^<>]+?)\s*,\s*([^<>]+?)\s*,\s*([^<>]+?)\s*>')


def parse_args():
    parser = argparse.ArgumentParser(description="Flowchart Triplet Extraction for MiniCPM-V-4_5")
    parser.add_argument("--base-model-path", type=str,
                        default="<MODEL_ROOT>/MiniCPM-V-4_5")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to the LoRA adapter")
    parser.add_argument("--test-json", type=str, default=None)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--image-ext", type=str, default=".jpeg")
    parser.add_argument("--backend", type=str, default="local", choices=["local"],
                        help="MiniCPM only supports the local backend (no vLLM path)")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    return parser.parse_args()


def parse_triplets_output(output_text):
    triplets = []
    if not output_text:
        return triplets
    for m in _TRIPLET_RE.finditer(output_text):
        src, cond, end = (p.strip() for p in m.groups())
        if src and end:
            triplets.append({"source": src, "condition": cond, "end": end})
    return triplets




def worker_process(gpu_id, task_queue, result_queue, args_dict, prompt):
    """Single-GPU worker: load MiniCPM-V + LoRA and process tasks in a loop."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    print(f"[GPU {gpu_id}] Worker started")

    try:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[GPU {gpu_id}] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args_dict["base_model_path"], trust_remote_code=True
        )

        print(f"[GPU {gpu_id}] Loading model (dtype={dtype})...")
        model = AutoModel.from_pretrained(
            args_dict["base_model_path"],
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        model = model.to(device)

        if args_dict["lora_path"]:
            print(f"[GPU {gpu_id}] Loading LoRA: {args_dict['lora_path']}")
            model = PeftModel.from_pretrained(
                model, args_dict["lora_path"], device_map=device
            )
        else:
            print(f"[GPU {gpu_id}] Using base model without LoRA")

        model.eval()
        print(f"[GPU {gpu_id}] Model ready on {device}")

        while True:
            task = task_queue.get()
            if task is None:
                break
            image_name, image_id, idx, total = task

            if not Path(image_name).suffix:
                image_path = os.path.join(
                    args_dict["image_dir"], image_name + args_dict["image_ext"]
                )
            else:
                image_path = os.path.join(args_dict["image_dir"], image_name)

            if not os.path.exists(image_path):
                print(f"[GPU {gpu_id}] Warning: missing {image_path}")
                result_queue.put((image_name, [], ""))
                continue

            print(f"[GPU {gpu_id}] [{idx}/{total}] {image_name}")

            try:
                img = Image.open(image_path).convert("RGB")
                msgs = [{"role": "user", "content": [img, prompt]}]
                with torch.no_grad():
                    output_text = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=False,
                        max_new_tokens=args_dict["max_new_tokens"],
                    )
                if isinstance(output_text, (list, tuple)):
                    output_text = output_text[0]
            except Exception as e:
                print(f"[GPU {gpu_id}] Error on {image_path}: {e}")
                import traceback
                traceback.print_exc()
                output_text = ""

            triplets = parse_triplets_output(output_text)
            result_queue.put((image_name, triplets, output_text))

            if idx <= 3:
                print(f"[GPU {gpu_id}] Output[:200]: {str(output_text)[:200]}")
                print(f"[GPU {gpu_id}] Parsed: {triplets}")

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker fatal: {e}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()

    print("=" * 60)
    print("MiniCPM-V-4_5 Triplet Extraction")
    print("=" * 60)
    print(f"Base Model : {args.base_model_path}")
    print(f"LoRA Path  : {args.lora_path or 'None'}")
    print(f"Test JSON  : {args.test_json or '(scan dir)'}")
    print(f"Image Dir  : {args.image_dir}")
    print(f"Output JSON: {args.output_json}")
    print(f"Max Tokens : {args.max_new_tokens}")
    print(f"Image Ext  : {args.image_ext}")
    print("=" * 60)

    if not args.gpus:
        print("Error: --gpus required, e.g. --gpus 0,1,2,3")
        return
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    print(f"GPUs: {gpu_ids}")

    if not os.path.exists(args.image_dir):
        print(f"Error: image dir not found: {args.image_dir}")
        return

    test_mapping = {}
    if args.test_json:
        if not os.path.exists(args.test_json):
            print(f"Error: test json not found: {args.test_json}")
            return
        with open(args.test_json, "r", encoding="utf-8") as f:
            content = f.read().strip()
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                test_mapping = data
            elif isinstance(data, list):
                test_mapping = {
                    item.get("image") or item.get("image_name") or item.get("filename", ""): item
                    for item in data
                }
            else:
                test_mapping = data
        except json.JSONDecodeError:
            test_mapping = {}
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (
                    record.get("image")
                    or record.get("image_name")
                    or record.get("filename")
                    or record.get("file_name", "")
                )
                test_mapping[key] = record
        print(f"Loaded {len(test_mapping)} items from JSON")
    else:
        p = Path(args.image_dir)
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]:
            for f in p.glob(f"*{ext}"):
                test_mapping[f.name] = ""
            for f in p.glob(f"*{ext.upper()}"):
                test_mapping[f.name] = ""
        if not test_mapping:
            print(f"Error: no images in {args.image_dir}")
            return
        print(f"Found {len(test_mapping)} images")

    prompt = get_e2e_prompt(with_image_tag=False)
    start = time.time()

    mp.set_start_method("spawn", force=True)
    task_queue = Queue()
    result_queue = Queue()

    total_samples = len(test_mapping)
    for idx, (image_name, _) in enumerate(test_mapping.items(), 1):
        image_id = Path(image_name).stem
        task_queue.put((image_name, image_id, idx, total_samples))
    wpg = args.workers_per_gpu
    total_workers = len(gpu_ids) * wpg
    for _ in range(total_workers):
        task_queue.put(None)

    args_dict = {
        "base_model_path": args.base_model_path,
        "lora_path": args.lora_path,
        "image_dir": args.image_dir,
        "max_new_tokens": args.max_new_tokens,
        "image_ext": args.image_ext,
    }

    processes = []
    for gpu_id in gpu_ids:
        for _ in range(wpg):
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, task_queue, result_queue, args_dict, prompt),
            )
            p.start()
            processes.append(p)

    results, raw_outputs = {}, {}
    completed = 0
    for _ in range(total_samples):
        image_name, triplets, raw = result_queue.get()
        results[image_name] = triplets
        raw_outputs[image_name] = raw
        completed += 1
        if completed % 10 == 0 or completed == total_samples:
            elapsed = time.time() - start
            speed = completed / elapsed if elapsed > 0 else 0
            print(f"Progress: {completed}/{total_samples} ({completed*100/total_samples:.1f}%) | {speed:.2f} imgs/s")

    for p in processes:
        p.join()

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    raw_path = os.path.splitext(args.output_json)[0] + "_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_outputs, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start
    print("=" * 60)
    print(f"Done -> {args.output_json}")
    print(f"Raw  -> {raw_path}")
    print(f"Total {len(results)} imgs | {total_time:.1f}s | {len(results)/total_time:.2f} imgs/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
