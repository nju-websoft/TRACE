"""FlowLearn inference with tool calling on a graph built from extracted triplets.

For each test image we ask the 6 standard FlowLearn sub-tasks
(Arrow_AtoB true/false, Arrow_betweenAB true/false, Num_Nodes, Num_Arrows)
and let the model decide whether to call graph tools.
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

from prompts import (
    flowlearn_isTrueFalse_AtoB,
    flowlearn_isTrueFalse_betweenAB,
    flowlearn_num_nodes_prompt,
    flowlearn_num_arrows_prompt,
)
from tool_runner import run_multi_gpu_tools


DEFAULT_TEST_JSON = "<REPO_ROOT>/QA/textflow_subset_testjson/flowlearn_test.json"
DEFAULT_IMAGE_DIR = "<DATA_ROOT>/flowlearn/mermaid_word/jpeg"


def _load_triplets(triplets_dir: str, image_stem: str) -> List[Dict]:
    p = os.path.join(triplets_dir, image_stem, "arrow_triplets.json")
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return []


def build_questions_for_image(sample: dict):
    """Return list of (sub_key, prompt, gt_value, extras)."""
    out = []
    a2b = sample.get("Arrow_AtoB", {})
    if "true" in a2b:
        a, b = a2b["true"]["a"], a2b["true"]["b"]
        out.append(("Flowchart-isTrue-AtoB",
                    flowlearn_isTrueFalse_AtoB(a, b),
                    "yes", {"a": a, "b": b}))
    if "false" in a2b:
        a, b = a2b["false"]["a"], a2b["false"]["b"]
        out.append(("Flowchart-isFalse-AtoB",
                    flowlearn_isTrueFalse_AtoB(a, b),
                    "no", {"a": a, "b": b}))
    bet = sample.get("Arrow_betweenAB", {})
    if "true" in bet:
        a, b = bet["true"]["a"], bet["true"]["b"]
        out.append(("Flowchart-isTrue-betweenAB",
                    flowlearn_isTrueFalse_betweenAB(a, b),
                    "yes", {"a": a, "b": b}))
    if "false" in bet:
        a, b = bet["false"]["a"], bet["false"]["b"]
        out.append(("Flowchart-isFalse-betweenAB",
                    flowlearn_isTrueFalse_betweenAB(a, b),
                    "no", {"a": a, "b": b}))
    if "Num_Nodes" in sample:
        out.append(("Num_Nodes",
                    flowlearn_num_nodes_prompt(),
                    str(sample["Num_Nodes"]), {}))
    if "Num_Arrows" in sample:
        out.append(("Num_Arrows",
                    flowlearn_num_arrows_prompt(),
                    str(sample["Num_Arrows"]), {}))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    p.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    p.add_argument("--triplets-dir", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--adapter-path", default="")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-new-tokens", type=int, default=64)
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
    plan: dict = {}
    skipped = 0
    n_with_triplets = 0
    for k in keys:
        img_path = os.path.join(args.image_dir, k)  # k already includes ".jpeg"
        if not os.path.exists(img_path):
            print(f"[warn] missing image: {img_path}")
            continue
        image_stem = os.path.splitext(k)[0]
        triplets = _load_triplets(args.triplets_dir, image_stem)
        if triplets:
            n_with_triplets += 1
        prior = existing.get(k, {})
        for sub_key, prompt, gt_val, extras in build_questions_for_image(gt[k]):
            plan[(k, sub_key)] = {"gt": gt_val, "extras": extras, "prompt": prompt}
            if sub_key in prior and prior[sub_key].get("predicted"):
                skipped += 1
                continue
            tasks.append({
                "image_path": img_path,
                "prompt": prompt,
                "out_key": (k, sub_key),
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
        k, sub_key = r["out_key"]
        if k not in out:
            out[k] = {}
        meta = plan[(k, sub_key)]
        out[k][sub_key] = {
            "Q": meta["prompt"],
            "extras": meta["extras"],
            "gt": meta["gt"],
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
