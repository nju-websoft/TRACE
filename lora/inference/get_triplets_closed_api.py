"""
Closed-model API inference: BPMN / any flowchart-like dataset, triplet extraction (whole-image).

The output JSON format is identical to get_triplets_sft_e2e.py:
  {"<image_name>": [{"source": ..., "condition": ..., "end": ...}, ...]}

Supports 2 backends:
  --backend openai  (gpt-4o)        env: OPENAI_API_KEY
  --backend zai     (glm-4.5v, Z.ai/Zhipu) env: ZAI_API_KEY (or ZHIPU_API_KEY)

resume: if output-json already exists, only run the missing images.

Example usage:
  OPENAI_API_KEY=sk-... python -m lora.inference.get_triplets_closed_api \
      --backend openai --model gpt-4o \
      --image-dir <DATA_ROOT>/bpmn/new_test_images \
      --output-json /tmp/bpmn_gpt4o.json \
      --max-new-tokens 8192

  ZAI_API_KEY=...    python -m lora.inference.get_triplets_closed_api --backend zai ...
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image


_TRIPLET_RE = re.compile(r'<\s*([^<>]+?)\s*,\s*([^<>]+?)\s*,\s*([^<>]+?)\s*>')


def generate_prompt() -> str:
    """Triplet-extraction prompt identical to get_triplets_sft_e2e.py (includes partOf)."""
    return (
        "Please describe all the information in the flowchart image in the form of triplets: "
        "For each arrow labeled with X directed from node A to node B, output a triplet: <A, X, B>; "
        "For each unlabeled arrow directed from node A to node B, output a triplet: <A, connectedTo, B>; "
        "For each node A fully inside node group B, output a triplet <A, partOf, B>.\n\n"
        "The following triplets are example outputs:\n"
        "<Food Packaging Improvement, secures, Quantum Computing Integration>\n"
        "<Graphene Production, catalyzes, Nanosensors>\n"
        "<Graphene Production, partOf, Drug Delivery Systems>\n"
        "<Nanocomposites, connectedTo, Spectroscopy>\n"
        "<Nanodevice Fabrication, connectedTo, Graphene Production>\n"
        "<Nanosensors, connectedTo, Nanocomposites>\n"
        "<Spectroscopy, educates, Nanodevice Fabrication>"
    )


def parse_triplets_output(text: str):
    triplets = []
    if not text:
        return triplets
    for m in _TRIPLET_RE.finditer(text):
        src, cond, end = (p.strip() for p in m.groups())
        if src and end:
            triplets.append({"source": src, "condition": cond, "end": end})
    return triplets


def encode_image_b64(image_path: str, max_pixels: int | None = None) -> tuple[str, str]:
    """Read an image, returning (base64-string, mime_type)."""
    img = Image.open(image_path).convert("RGB")
    if max_pixels is not None and img.width * img.height > max_pixels:
        # Scale (keeping aspect ratio) to max_pixels
        ratio = (max_pixels / (img.width * img.height)) ** 0.5
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


# ─── Backend: OpenAI ─────────────────────────────────────────────────────────

async def _call_openai(client, model, prompt, image_path, max_tokens):
    b64, mime = encode_image_b64(image_path)
    # GPT-5 / o1 / o3 are reasoning models by default: reasoning first, then content;
    # triplet task needs no reasoning; force effort=minimal to save tokens + speed up.
    # non-reasoning models (e.g. gpt-4o) ignore this parameter.
    resp = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        max_tokens=max_tokens,
        extra_body={"reasoning": {"effort": "minimal"}},
    )
    return resp.choices[0].message.content or ""


# ─── Backend: Z.ai (GLM-4.5V) — OpenAI-compatible API ───────────────────────────────

async def _call_zai(client, model, prompt, image_path, max_tokens):
    b64, mime = encode_image_b64(image_path)
    # GLM-4.6V is a thinking model by default: reasoning first, then content;
    # triplet task needs no reasoning; disable it to save tokens + speed up.
    resp = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return resp.choices[0].message.content or ""


# ─── Main async pipeline ─────────────────────────────────────────────────────

async def run(args):
    prompt = generate_prompt()

    image_dir = args.image_dir
    image_ext = args.image_ext.lower()
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(image_ext)
    ])
    print(f"Found {len(image_files)} images in {image_dir}")

    if args.test_json and os.path.exists(args.test_json):
        with open(args.test_json, "r", encoding="utf-8") as f:
            tj = json.load(f)
        keep = set()
        if isinstance(tj, dict):
            for k in tj.keys():
                # key may or may not include the extension
                base = os.path.splitext(k)[0]
                keep.add(k)
                keep.add(base + image_ext)
                keep.add(base + ".png")
                keep.add(base + ".jpg")
                keep.add(base + ".jpeg")
        elif isinstance(tj, list):
            for item in tj:
                if isinstance(item, dict):
                    for fk in ("image", "image_name", "key", "filename"):
                        if fk in item:
                            v = item[fk]
                            base = os.path.splitext(os.path.basename(v))[0]
                            keep.add(os.path.basename(v))
                            keep.add(base + image_ext)
                            break
                elif isinstance(item, str):
                    base = os.path.splitext(os.path.basename(item))[0]
                    keep.add(os.path.basename(item))
                    keep.add(base + image_ext)
        before = len(image_files)
        image_files = [f for f in image_files if f in keep or os.path.splitext(f)[0] in {os.path.splitext(k)[0] for k in keep}]
        print(f"After --test-json filter: {len(image_files)}/{before}")

    if args.max_images is not None:
        image_files = image_files[:args.max_images]
        print(f"Limited to first {len(image_files)} images (--max-images)")

    # resume: load existing
    existing = {}
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
            print(f"Existing predictions: {len(existing)} (resume)")
        except Exception as e:
            print(f"Warning: couldn't load existing output, restarting: {e}")
            existing = {}

    raw_outputs = {}
    raw_path = os.path.splitext(args.output_json)[0] + "_raw.json"
    if os.path.exists(raw_path):
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_outputs = json.load(f)
        except Exception:
            raw_outputs = {}

    # resume: skip when key is in existing and (triplets non-empty OR raw non-empty)
    # empty raw counts as a failure and needs a re-run
    def _is_complete(fname):
        if fname not in existing:
            return False
        if existing.get(fname):  # has triplets
            return True
        if raw_outputs.get(fname):  # has raw content (counts even if no triplets were parsed)
            return True
        return False

    todo = [f for f in image_files if not _is_complete(f)]
    skipped_empty = sum(1 for f in image_files if f in existing and not _is_complete(f))
    print(f"To process: {len(todo)} (skip {len(image_files) - len(todo)}; "
          f"of which empty-rerun: {skipped_empty})")

    if not todo:
        print("Nothing to do. ✔")
        return

    # Init client
    backend = args.backend
    if backend == "openai":
        from openai import AsyncOpenAI
        api_key = (os.environ.get("OPENROUTER_API_KEY")
                   or os.environ.get("OPENAI_API_KEY")
                   or args.api_key)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY / OPENROUTER_API_KEY not set")
        # OpenRouter / any OpenAI-compatible proxy is selected via --api-base
        client_kwargs = {"api_key": api_key}
        if args.api_base:
            client_kwargs["base_url"] = args.api_base
        client = AsyncOpenAI(**client_kwargs)
        call = _call_openai
    elif backend == "zai":
        from openai import AsyncOpenAI
        api_key = (os.environ.get("ZAI_API_KEY")
                   or os.environ.get("ZHIPU_API_KEY")
                   or args.api_key)
        if not api_key:
            raise RuntimeError("ZAI_API_KEY / ZHIPU_API_KEY not set")
        # Z.ai overseas API; for Zhipu (CN) use https://open.bigmodel.cn/api/paas/v4/
        base_url = args.api_base or "https://api.z.ai/api/paas/v4/"
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        call = _call_zai
    else:
        raise ValueError(f"Unknown backend: {backend}")

    sem = asyncio.Semaphore(args.max_concurrent)
    results = dict(existing)
    completed = 0
    start_ts = time.time()
    lock = asyncio.Lock()

    async def worker(fname: str):
        nonlocal completed
        img_path = os.path.join(image_dir, fname)
        async with sem:
            for attempt in range(args.max_retries + 1):
                try:
                    text = await call(client, args.model, prompt, img_path, args.max_new_tokens)
                    break
                except Exception as e:
                    if attempt >= args.max_retries:
                        print(f"  ✗ {fname}: FAILED after {attempt+1} attempts: {e}")
                        text = ""
                        break
                    await asyncio.sleep(2 ** attempt)
            triplets = parse_triplets_output(text)
            async with lock:
                results[fname] = triplets
                raw_outputs[fname] = text
                completed += 1
                if completed % args.save_every == 0 or completed == len(todo):
                    with open(args.output_json, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    with open(raw_path, "w", encoding="utf-8") as f:
                        json.dump(raw_outputs, f, ensure_ascii=False, indent=2)
            elapsed = time.time() - start_ts
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"  [{completed}/{len(todo)}] {fname} → {len(triplets)} triplets  ({rate:.2f} img/s)")

    tasks = [asyncio.create_task(worker(f)) for f in todo]
    await asyncio.gather(*tasks)

    # final save
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_outputs, f, ensure_ascii=False, indent=2)

    total = time.time() - start_ts
    print(f"\n✔ Done: {len(results)} images | {total:.1f}s | rate {len(todo)/total:.2f} img/s")
    print(f"  output: {args.output_json}")
    print(f"  raw:    {raw_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["openai", "zai"])
    p.add_argument("--model", required=True,
                   help="model id, e.g. gpt-4o / glm-4.5v")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--image-ext", default=".png")
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-concurrent", type=int, default=10,
                   help="Number of concurrent requests (OpenAI/Z.ai are rate-limited; default 10)")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--save-every", type=int, default=5,
                   help="Save intermediate results every N images")
    p.add_argument("--max-images", type=int, default=None,
                   help="Process only the first N images (for testing)")
    p.add_argument("--test-json", type=str, default=None,
                   help="Test set JSON (key=image_name, or a list of {image/key}); used to filter image_dir")
    p.add_argument("--api-key", default=None,
                   help="Override the environment variable")
    p.add_argument("--api-base", default=None,
                   help="Custom base URL: OpenRouter=https://openrouter.ai/api/v1 ; Zhipu (CN)=https://open.bigmodel.cn/api/paas/v4/")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Backend: {args.backend}  Model: {args.model}")
    print(f"Image dir: {args.image_dir}")
    print(f"Output:    {args.output_json}")
    print(f"Concurrent: {args.max_concurrent}  Max tokens: {args.max_new_tokens}")
    asyncio.run(run(args))
