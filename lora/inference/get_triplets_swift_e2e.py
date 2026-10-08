"""
get_triplets_swift_e2e.py

Triplet-extraction inference script — generic HuggingFace VLM version (not Qwen-specific).
For models fine-tuned via ms-swift such as Gemma3-4B-IT / LLaVa-v1.6-Mistral-7B-hf,
the resulting LoRA adapter. The CLI is fully aligned with lora.inference.get_triplets_sft_e2e:
  --base-model-path / --lora-path / --image-dir / --output-json / ...

Key differences from the v2 (Qwen) version:
  1. No longer depends on qwen_vl_utils.process_vision_info
  2. Feed images directly to AutoProcessor as PIL.Image
  3. Message format uses a {"type": "image"} placeholder + processor(images=[pil_img])
  4. Multi-GPU / multi-worker behavior matches v2

The output JSON structure is identical to v2 and can be scored directly by the eval scripts.
"""
import os
import json
import time
import argparse
from lora.prompt import get_e2e_prompt
import re
import multiprocessing as mp
from multiprocessing import Queue
from pathlib import Path
from PIL import Image


# ================================================================
# CLI
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Flowchart Triplet Extraction (Swift / Generic HF VLM)"
    )
    parser.add_argument("--base-model-path", type=str, required=True,
                        help="Path to the base model, e.g. .../gemma-3-4b-it")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to the LoRA adapter; omit to use the base model only")
    parser.add_argument("--test-json", type=str, default=None,
                        help="Path to the test set JSON/JSONL (optional)")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Source image directory")
    parser.add_argument("--output-json", type=str, required=True,
                        help="Output JSON path (dict: filename -> [triplets])")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--gpus", type=str, default=None,
                        help="GPU IDs, comma-separated, e.g. 0,1,2,3")
    parser.add_argument("--image-ext", type=str, default=".jpeg",
                        help="Extension used when completing stem -> filename")
    parser.add_argument("--backend", type=str, default="local",
                        choices=["local"])
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    return parser.parse_args()


# ================================================================
# Output parsing (identical to v2)
# ================================================================
def parse_triplets_output(output_text):
    triplets = []
    pattern = r"<\s*([^,>]+?)\s*,\s*([^,>]+?)\s*,\s*([^,>]+?)\s*>"
    for m in re.finditer(pattern, output_text):
        start, cond, end = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if start and cond and end:
            triplets.append({"source": start, "condition": cond, "end": end})
    return triplets


# ================================================================
# Prompt (identical to v2)
# ================================================================


# ================================================================
def worker_process(gpu_id, task_queue, result_queue, args_dict, prompt):
    """Subprocess: pin CUDA_VISIBLE_DEVICES first, then import torch / transformers / peft."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    # the literal <image> in the template is for Qwen-VL in prompt.py; via apply_chat_template
    # generic HF VLMs (LLaVa-Next / Gemma3 / etc.) insert another <image> themselves; in the prompt
    # another one would cause "2 image_tokens vs 1 image" -> StopIteration (and the exception
    # its str() is empty, so the log shows "Error on ...:" followed by a blank line).
    prompt = prompt.replace("<image>", "").lstrip()

    device = f"cuda:0"
    print(f"[GPU {gpu_id}] Worker started")

    try:
        print(f"[GPU {gpu_id}] Loading processor ...")
        processor = AutoProcessor.from_pretrained(
            args_dict["base_model_path"], trust_remote_code=True
        )

        dtype = torch.bfloat16
        print(f"[GPU {gpu_id}] Loading model (dtype={dtype}) ...")
        model = AutoModelForImageTextToText.from_pretrained(
            args_dict["base_model_path"],
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )

        if args_dict.get("lora_path"):
            print(f"[GPU {gpu_id}] Loading LoRA from {args_dict['lora_path']} ...")
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
                print(f"[GPU {gpu_id}] Warning: Image not found: {image_path}")
                result_queue.put((image_name, [], ""))
                continue

            print(f"[GPU {gpu_id}] Processing [{idx}/{total}]: {image_name}")

            output_text = ""
            try:
                pil_img = Image.open(image_path).convert("RGB")

                # Generic HF VLM message format: image placeholder + text
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # feed the PIL.Image directly to the processor -- supported by Gemma3 / LLaVa-Next
                inputs = processor(
                    text=[text],
                    images=[pil_img],
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args_dict["max_new_tokens"],
                        do_sample=False,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            except Exception as e:
                print(f"[GPU {gpu_id}] Error on {image_path}: {e}")
                output_text = ""

            triplets = parse_triplets_output(output_text)
            result_queue.put((image_name, triplets, output_text))
            if idx <= 3 or idx % 50 == 0:
                print(f"[GPU {gpu_id}] Output[:200]: {output_text[:200]}")
                print(f"[GPU {gpu_id}] Parsed: {triplets}")

    except Exception as e:
        import traceback
        print(f"[GPU {gpu_id}] Worker fatal error: {e}")
        traceback.print_exc()


# ================================================================
# main
# ================================================================
def main():
    args = parse_args()
    print("=" * 60)
    print("Flowchart Triplet Extraction Pipeline (Swift / Generic VLM)")
    print("=" * 60)
    print(f"Base Model : {args.base_model_path}")
    print(f"LoRA Path  : {args.lora_path or 'None (base only)'}")
    print(f"Test JSON  : {args.test_json or 'None (scan image dir)'}")
    print(f"Image Dir  : {args.image_dir}")
    print(f"Output JSON: {args.output_json}")
    print(f"Max Tokens : {args.max_new_tokens}")
    print(f"Image Ext  : {args.image_ext}")
    print("=" * 60 + "\n")

    if not args.gpus:
        print("Error: --gpus is required, e.g. --gpus 0,1")
        return
    actual_gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    print(f"Using GPUs: {actual_gpu_ids} (Total: {len(actual_gpu_ids)})")

    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return

    # Load the test set index
    test_mapping = {}
    if args.test_json:
        if not os.path.exists(args.test_json):
            print(f"Error: Test JSON not found: {args.test_json}")
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
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (record.get("image") or record.get("image_name")
                       or record.get("filename") or record.get("file_name", ""))
                test_mapping[key] = record
        print(f"Loaded {len(test_mapping)} images from JSON\n")
    else:
        print("No test JSON provided, scanning image directory ...")
        image_dir_path = Path(args.image_dir)
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]:
            for p in image_dir_path.glob(f"*{ext}"):
                test_mapping[p.name] = ""
            for p in image_dir_path.glob(f"*{ext.upper()}"):
                test_mapping[p.name] = ""
        if not test_mapping:
            print(f"Error: No images found in {args.image_dir}")
            return
        print(f"Found {len(test_mapping)} images\n")

    prompt = get_e2e_prompt()
    start_time = time.time()

    mp.set_start_method("spawn", force=True)
    task_queue = Queue()
    result_queue = Queue()

    total_samples = len(test_mapping)
    for idx, (image_name, _) in enumerate(test_mapping.items(), 1):
        image_id = Path(image_name).stem
        task_queue.put((image_name, image_id, idx, total_samples))
    wpg = args.workers_per_gpu
    total_workers = len(actual_gpu_ids) * wpg
    for _ in range(total_workers):
        task_queue.put(None)

    args_dict = {
        "base_model_path": args.base_model_path,
        "lora_path": args.lora_path,
        "image_dir": args.image_dir,
        "max_new_tokens": args.max_new_tokens,
        "image_ext": args.image_ext,
    }

    print(f"Starting {total_workers} workers ({wpg} per GPU) ...\n")
    processes = []
    for gpu_id in actual_gpu_ids:
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
            elapsed = time.time() - start_time
            speed = completed / elapsed if elapsed > 0 else 0
            print(f"Progress: {completed}/{total_samples} "
                  f"({completed*100/total_samples:.1f}%) | {speed:.2f} imgs/s")

    for p in processes:
        p.join()

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    raw_output_path = os.path.splitext(args.output_json)[0] + "_raw.json"
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(raw_outputs, f, ensure_ascii=False, indent=2)
    print(f"Raw outputs saved -> {raw_output_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✔ Done! Results -> {args.output_json}")
    print(f"Total: {len(results)} images | Time: {total_time:.2f}s | "
          f"Speed: {len(results)/total_time:.2f} imgs/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
