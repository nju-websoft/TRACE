import json
import os
import re
import argparse
from lora.prompt import get_e2e_prompt
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Queue
import time


_TRIPLET_RE = re.compile(r'<\s*([^<>]+?)\s*,\s*([^<>]+?)\s*,\s*([^<>]+?)\s*>')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flowchart Triplet Extraction with Multi-GPU Support")
    parser.add_argument("--base-model-path", type=str,
                        default="<MODEL_ROOT>/Qwen3-VL-4B-Instruct",
                        help="Path to the base model")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to the LoRA adapter; omit to use the base model only")
    parser.add_argument("--test-json", type=str, default=None,
                        help="Path to the test set JSON (optional)")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Image directory")
    parser.add_argument("--output-json", type=str, required=True,
                        help="Path to the output JSON")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max number of generated tokens")
    parser.add_argument("--gpus", type=str, default=None,
                        help="GPU IDs to use, comma-separated, e.g. '0,1,2,3'")
    parser.add_argument("--image-ext", type=str, default=".jpeg",
                        help="Image file extension (e.g. .jpeg, .png, .jpg)")
    parser.add_argument("--backend", type=str, default="local", choices=["local"],
                        help="Inference backend: local (multi-GPU)")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Workers to launch per GPU")
    return parser.parse_args()


def parse_triplets_output(output_text):
    """Parse the model output triplets, extracting source, end, condition.
    Loose matching: recognize <A, rel, B> anywhere, tolerating '1. <...>', '- <...>', markdown, surrounding preamble, etc.
    """
    triplets = []
    if not output_text:
        return triplets
    for m in _TRIPLET_RE.finditer(output_text):
        src, cond, end = (p.strip() for p in m.groups())
        if src and end:
            triplets.append({
                "source_node": src,
                "condition": cond,
                "target_node": end,
            })
    return triplets





def worker_process(gpu_id, task_queue, result_queue, args_dict, prompt):
    """
    Worker process for one GPU.
    Important: set CUDA_VISIBLE_DEVICES first, then import torch/transformers/peft,
    Ensure this process sees only its own GPU and accesses it uniformly as cuda:0.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"  # within this process the only visible GPU is cuda:0

    # Lazy import, so the env vars set above take effect first
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    print(f"[GPU {gpu_id}] Worker started, visible device: {device}")

    try:
        # Load the processor (CPU, lightweight)
        print(f"[GPU {gpu_id}] Loading processor (CPU)...")
        processor = AutoProcessor.from_pretrained(args_dict["base_model_path"])

        # Load the model onto this GPU
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[GPU {gpu_id}] Loading model (dtype={dtype})...")
        model = AutoModelForImageTextToText.from_pretrained(
            args_dict["base_model_path"],
            torch_dtype=dtype,
            device_map=device
        )

        if args_dict["lora_path"]:
            print(f"[GPU {gpu_id}] Loading LoRA from {args_dict['lora_path']}...")
            model = PeftModel.from_pretrained(model, args_dict["lora_path"], device_map=device)
        else:
            print(f"[GPU {gpu_id}] Using base model without LoRA")

        model.eval()
        print(f"[GPU {gpu_id}] Model ready on {device}")

        # keep processing tasks
        while True:
            task = task_queue.get()
            if task is None:
                break

            image_name, image_id, idx, total = task

            # Build the image path
            if not Path(image_name).suffix:
                image_path = os.path.join(
                    args_dict["image_dir"], image_name + args_dict["image_ext"])
            else:
                image_path = os.path.join(args_dict["image_dir"], image_name)

            if not os.path.exists(image_path):
                print(f"[GPU {gpu_id}] Warning: Image not found: {image_path}")
                result_queue.put((image_name, [], ""))
                continue

            print(f"[GPU {gpu_id}] Processing [{idx}/{total}]: {image_name}")

            # Inference
            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args_dict["max_new_tokens"],
                        do_sample=False
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing {image_path}: {e}")
                output_text = ""

            triplets = parse_triplets_output(output_text)
            result_queue.put((image_name, triplets, output_text))

            if idx <= 3:
                print(f"[GPU {gpu_id}] Output: {output_text[:200]}...")
                print(f"[GPU {gpu_id}] Parsed triplets: {triplets}")

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker error: {e}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()

    print(f"{'='*60}")
    print(f"Flowchart Triplet Extraction Pipeline")
    print(f"{'='*60}")
    print(f"Base Model : {args.base_model_path}")
    print(f"LoRA Path  : {args.lora_path or 'None (base model only)'}")
    print(f"Test JSON  : {args.test_json or 'None (scan image dir)'}")
    print(f"Image Dir  : {args.image_dir}")
    print(f"Output JSON: {args.output_json}")
    print(f"Max Tokens : {args.max_new_tokens}")
    print(f"Image Ext  : {args.image_ext}")
    print(f"{'='*60}\n")

    # Determine the GPU list
    actual_gpu_ids = []
    if args.gpus:
        actual_gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        print("Error: --gpus must be specified, e.g. --gpus 0,1,2,3")
        return
    print(f"Using GPUs: {actual_gpu_ids} (Total: {len(actual_gpu_ids)})")

    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return

    # Load the test set
    test_mapping = {}
    if args.test_json:
        if not os.path.exists(args.test_json):
            print(f"Error: Test JSON not found: {args.test_json}")
            return
        with open(args.test_json, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                test_mapping = data
            elif isinstance(data, list):
                test_mapping = {item.get("image") or item.get("image_name") or item.get("filename", ""): item for item in data}
            else:
                test_mapping = data
        except json.JSONDecodeError:
            # JSONL format: one JSON object per line
            test_mapping = {}
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = record.get("image") or record.get("image_name") or record.get("filename") or record.get("file_name", "")
                test_mapping[key] = record
        print(f"Loaded {len(test_mapping)} images from JSON\n")
    else:
        print("No test JSON provided, scanning image directory...")
        image_dir_path = Path(args.image_dir)
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
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

    mp.set_start_method('spawn', force=True)
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

    print(f"Starting {total_workers} worker processes ({wpg} per GPU)...\n")
    processes = []
    for gpu_id in actual_gpu_ids:
        for _ in range(wpg):
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, task_queue, result_queue, args_dict, prompt)
            )
            p.start()
            processes.append(p)

    results = {}
    raw_outputs = {}
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

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # also save a raw-output sidecar for later re-parsing (does not affect the evaluator reading output_json)
    raw_output_path = os.path.splitext(args.output_json)[0] + "_raw.json"
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        json.dump(raw_outputs, f, ensure_ascii=False, indent=2)
    print(f"Raw outputs saved -> {raw_output_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✔ Done! Results -> {args.output_json}")
    print(f"Total: {len(results)} images | Time: {total_time:.2f}s | "
          f"Speed: {len(results)/total_time:.2f} imgs/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()