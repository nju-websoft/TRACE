"""Shared tool-calling QA runner: multi-GPU local Qwen-VL with graph tools.

Each task is a dict {"image_path": str, "prompt": str, "out_key": tuple,
"triplets": list}. The per-task tool-call loop:
  1. apply_chat_template(messages, tools=TOOL_SCHEMAS) → generate
  2. parse <tool_call>{...}</tool_call> blocks from the model's output
  3. dispatch each call against a FlowGraph built from the task's triplets
  4. append role=tool messages with the JSON results
  5. repeat until no tool calls or `max_steps` reached
The final assistant text (with any stray tool_call blocks stripped) is the answer.
"""

import os
import re
import json
import multiprocessing as mp
from typing import Dict, List

from graph_tools import FlowGraph, dispatch_tool, TOOL_SCHEMAS


TOOL_SYSTEM_PROMPT = (
    "You are an expert assistant for flowchart question answering.\n"
    "You have access to tools that query a graph extracted from the flowchart "
    "(nodes are text boxes; edges are arrows; some edges have a condition label).\n"
    "When a question is best answered with graph facts (counts, neighbors, "
    "edge existence, shortest path, what comes next/before, etc.), CALL TOOLS "
    "first, then answer.\n"
    "Node labels in the graph may differ slightly from the question's wording — "
    "use `find_node` to resolve ambiguous names before passing them to other tools.\n"
    "Use at most a handful of tool calls per question (no loops).\n"
    "When you are ready to answer, give a SHORT plain-text response. "
    "Do NOT wrap the final answer in <tool_call> tags or JSON."
)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str):
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        name = obj.get("name", "")
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "arguments": args})
    return calls


def strip_tool_calls(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


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
    max_steps: int,
    no_image: bool,
    fuzzy: float,
    system_prompt: str,
    out_queue,
):
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
    print(f"[GPU {gpu_id}] Loading model {base_model_path}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    if adapter_path:
        print(f"[GPU {gpu_id}] Loading LoRA adapter {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model_path)

    results = []
    image_cache = {}
    total = len(task_list)
    for i, task in enumerate(task_list, 1):
        img_path = task["image_path"]
        prompt = task["prompt"]
        out_key = task["out_key"]
        triplets = task.get("triplets", [])
        graph = FlowGraph(triplets, fuzzy=fuzzy)

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

            messages = [
                {"role": "system",
                 "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": user_content},
            ]

            if not no_image:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs, video_inputs = None, None

            tool_trace = []
            final_text = ""
            for step in range(max_steps + 1):
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    tools=TOOL_SCHEMAS,
                )
                if no_image:
                    inputs = processor(
                        text=[text_input], padding=True, return_tensors="pt"
                    ).to(device)
                else:
                    inputs = processor(
                        text=[text_input], images=image_inputs,
                        videos=video_inputs, padding=True, return_tensors="pt",
                    ).to(device)

                with torch.no_grad():
                    gen = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    )
                trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
                resp = processor.batch_decode(
                    trimmed, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                calls = parse_tool_calls(resp)
                if not calls or step == max_steps:
                    final_text = strip_tool_calls(resp)
                    break

                messages.append({"role": "assistant", "content": resp})
                for c in calls:
                    name = c["name"]
                    args = c["arguments"]
                    try:
                        result = dispatch_tool(graph, name, args)
                    except Exception as e:
                        result = {"error": str(e)}
                    tool_trace.append(
                        {"step": step, "name": name, "arguments": args,
                         "result": result}
                    )
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            results.append({
                "out_key": out_key,
                "predicted": final_text,
                "tool_trace": tool_trace,
                "ok": True,
            })
        except Exception as e:
            print(f"[GPU {gpu_id}] FAIL {out_key}: {e}", flush=True)
            results.append({"out_key": out_key, "predicted": "",
                            "tool_trace": [], "ok": False, "error": str(e)})

        if i % 50 == 0 or i == total:
            print(f"[GPU {gpu_id}] {i}/{total} done", flush=True)

        if len(image_cache) > 16:
            image_cache.clear()

    print(f"[GPU {gpu_id}] Worker finished: "
          f"{sum(r['ok'] for r in results)}/{total} ok", flush=True)
    out_queue.put(results)


def run_multi_gpu_tools(
    tasks: List[Dict],
    gpus: List[int],
    base_model_path: str,
    adapter_path: str,
    max_new_tokens: int = 256,
    max_steps: int = 5,
    no_image: bool = False,
    fuzzy: float = 0.85,
    system_prompt: str = TOOL_SYSTEM_PROMPT,
) -> List[Dict]:
    if not tasks:
        return []
    if not gpus:
        gpus = [0]
    chunks = _split_evenly(tasks, len(gpus))
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = []
    for gpu_id, chunk in zip(gpus, chunks):
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, chunk, base_model_path, adapter_path,
                  max_new_tokens, max_steps, no_image, fuzzy,
                  system_prompt, out_q),
        )
        p.start()
        procs.append(p)
    all_results = []
    for _ in procs:
        all_results.extend(out_q.get())
    for p in procs:
        p.join()
    return all_results
