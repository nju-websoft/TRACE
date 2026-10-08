"""FlowVQA inference with tool calling on a graph built from extracted triplets.

For each test question we let Qwen3-VL decide whether to query the graph
(via TOOL_SCHEMAS in `tool_runner.py`) before answering.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "QA"))

from tool_runner import run_multi_gpu_tools


DEFAULT_TEST_JSON = "<REPO_ROOT>/QA/textflow_subset_testjson/flowvqa_test.json"
DEFAULT_IMAGE_DIR = "<DATA_ROOT>/flowvqa/Data/A. Main Set Flowchart Images"


def _load_triplets(triplets_dir: str, image_stem: str) -> List[Dict]:
    p = os.path.join(triplets_dir, image_stem, "arrow_triplets.json")
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return []


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    p.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    p.add_argument("--triplets-dir", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--adapter-path", default="")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-image", action="store_true")
    p.add_argument("--fuzzy", type=float, default=0.85)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.test_json) as f:
        gt = json.load(f)
    keys = sorted(gt.keys())
    if args.limit > 0:
        keys = keys[: args.limit]

    existing: dict = {}
    if args.resume and os.path.exists(args.output_file):
        try:
            with open(args.output_file) as f:
                existing = json.load(f)
            print(f"[resume] loaded {len(existing)} prior images")
        except Exception as e:
            print(f"[resume] load failed: {e}")

    tasks: List[Dict] = []
    skipped = 0
    n_with_triplets = 0
    for k in keys:
        sample = gt[k]
        img_path = os.path.join(args.image_dir, f"{k}.png")
        if not os.path.exists(img_path):
            print(f"[warn] missing image: {img_path}")
            continue
        triplets = _load_triplets(args.triplets_dir, k)
        if triplets:
            n_with_triplets += 1
        prior = existing.get(k, {})
        for qid, item in sample.get("qa", {}).items():
            if qid in prior and prior[qid].get("predicted"):
                skipped += 1
                continue
            prompt = (
                "Answer the following question about the flowchart. "
                "Give a short, concise answer.\n\n"
                f"Question: {item['Q']}"
            )
            tasks.append({
                "image_path": img_path,
                "prompt": prompt,
                "out_key": (k, qid),
                "triplets": triplets,
            })

    print(f"[plan] images={len(keys)}  with_triplets={n_with_triplets}/{len(keys)}")
    print(f"[plan] tasks={len(tasks)}  resume_skipped={skipped}")
    print(f"[plan] adapter={'(none)' if not args.adapter_path else args.adapter_path}")

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    results = run_multi_gpu_tools(
        tasks, gpus=gpus,
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        max_steps=args.max_steps,
        no_image=args.no_image,
        fuzzy=args.fuzzy,
    )

    out: dict = existing
    for r in results:
        k, qid = r["out_key"]
        if k not in out:
            out[k] = {}
        item = gt[k]["qa"][qid]
        out[k][qid] = {
            "Q": item["Q"],
            "A1": item.get("A1", ""),
            "A2": item.get("A2", ""),
            "A3": item.get("A3", ""),
            "type": item.get("type", ""),
            "predicted": r["predicted"],
            "tool_trace": r.get("tool_trace", []),
            "ok": r["ok"],
        }

    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] saved → {args.output_file} "
          f"({sum(len(v) for v in out.values())} predictions)")


if __name__ == "__main__":
    main()
