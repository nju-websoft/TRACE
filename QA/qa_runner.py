"""Shared QA inference runner: multi-GPU local Qwen-VL with optional LoRA.

Each task is a dict {"image_path": str, "prompt": str, "out_key": Tuple[...]}.
Results are written to a single JSON: {image_id: {sub_key: {prompt, predicted, ...}}}.
"""

import os
import json
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Dict, List, Tuple


def _split_evenly(items: List, n: int) -> List[List]:
    chunks = [[] for _ in range(n)]
    for i, x in enumerate(items):
        chunks[i % n].append(x)
    return chunks


def _worker(
    gpu_id: int,
    task_list: List[Dict],
    base_model_path: str,
    adapter_path: str,
    max_new_tokens: int,
    system_prompt: str,
    no_image: bool,
    out_queue,
):
    """Run a chunk of tasks on one GPU and push results back via queue."""
    if not task_list:
        out_queue.put([])
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    device = "cuda:0"
    print(f"[GPU {gpu_id}] Loading model {base_model_path} ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    if adapter_path:
        print(f"[GPU {gpu_id}] Loading LoRA adapter {adapter_path} ...", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"[GPU {gpu_id}] Using BASE model (no adapter)", flush=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model_path)

    results = []
    image_cache: Dict[str, Image.Image] = {}
    total = len(task_list)
    for i, task in enumerate(task_list, 1):
        img_path = task["image_path"]
        prompt = task["prompt"]
        out_key = task["out_key"]

        try:
            user_content = []
            if not no_image:
                if img_path in image_cache:
                    pil = image_cache[img_path]
                else:
                    pil = Image.open(img_path).convert("RGB")
                    image_cache[img_path] = pil
                user_content.append({"type": "image", "image": pil})
            user_content.append({"type": "text", "text": prompt})

            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                })
            messages.append({"role": "user", "content": user_content})

            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if no_image:
                inputs = processor(
                    text=[text_input], padding=True, return_tensors="pt",
                ).to(device)
            else:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
            text_out = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            results.append({"out_key": out_key, "predicted": text_out, "ok": True})
        except Exception as e:
            print(f"[GPU {gpu_id}] [{i}/{total}] FAIL {img_path} {out_key}: {e}", flush=True)
            results.append({"out_key": out_key, "predicted": "", "ok": False, "error": str(e)})

        if i % 50 == 0 or i == total:
            print(f"[GPU {gpu_id}] {i}/{total} done", flush=True)

        # free image cache periodically — these are mid-size and can pile up
        if len(image_cache) > 16:
            image_cache.clear()

    print(f"[GPU {gpu_id}] Worker finished: {sum(r['ok'] for r in results)}/{total} ok", flush=True)
    out_queue.put(results)


def run_multi_gpu(
    tasks: List[Dict],
    gpus: List[int],
    base_model_path: str,
    adapter_path: str,
    max_new_tokens: int = 32,
    system_prompt: str = "",
    no_image: bool = False,
) -> List[Dict]:
    """Execute tasks across GPUs and return aggregated results.
    Order is NOT preserved; result entries carry their out_key so callers can route them."""
    if not tasks:
        return []
    if len(gpus) == 0:
        gpus = [0]

    chunks = _split_evenly(tasks, len(gpus))
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = []
    for gpu_id, chunk in zip(gpus, chunks):
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, chunk, base_model_path, adapter_path,
                  max_new_tokens, system_prompt, no_image, out_q),
        )
        p.start()
        procs.append(p)

    all_results = []
    for _ in procs:
        all_results.extend(out_q.get())
    for p in procs:
        p.join()
    return all_results


def parse_gpus(gpus_str: str) -> List[int]:
    return [int(x) for x in gpus_str.split(",") if x.strip()]
