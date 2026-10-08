"""Run FlowLearn QA on a Qwen-VL model — first 4 sub-tasks (official format).

Per image, 6 questions (FlowGen Appendix-E Yes/No prompt):
  - Flowchart-isTrue-AtoB        → expected answer "yes"   (uses Arrow_AtoB.true.{a,b})
  - Flowchart-isFalse-AtoB       → expected answer "no"    (uses Arrow_AtoB.false.{a,b})
  - Flowchart-isTrue-betweenAB   → expected answer "yes"   (uses Arrow_betweenAB.true.{a,b})
  - Flowchart-isFalse-betweenAB  → expected answer "no"    (uses Arrow_betweenAB.false.{a,b})
  - Num_Nodes                    → integer
  - Num_Arrows                   → integer

Statement phrasing follows FlowLearn's official `dataset_processing.py`;
prompt envelope and system prompt follow FlowGen (ICLR 2026, App. E).
"""

import os
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "QA"))

from prompts import (
    flowlearn_isTrueFalse_AtoB,
    flowlearn_isTrueFalse_betweenAB,
    flowlearn_num_nodes_prompt,
    flowlearn_num_arrows_prompt,
    format_triplet_block,
    SYSTEM_PROMPT,
)
from qa_runner import run_multi_gpu, parse_gpus


def _load_triplets(triplets_dir: str, image_stem: str):
    if not triplets_dir:
        return []
    p = os.path.join(triplets_dir, image_stem, "arrow_triplets.json")
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return []


DEFAULT_TEST_JSON = "<REPO_ROOT>/QA/textflow_subset_testjson/flowlearn_test.json"
DEFAULT_IMAGE_DIR = "<DATA_ROOT>/flowlearn/mermaid_word/jpeg"


def build_questions_for_image(sample: dict, triplet_block: str = ""):
    """Return list of (sub_key, prompt, gt_value, extras)."""
    out = []
    a2b = sample.get("Arrow_AtoB", {})
    if "true" in a2b:
        a, b = a2b["true"]["a"], a2b["true"]["b"]
        out.append(("Flowchart-isTrue-AtoB",
                    flowlearn_isTrueFalse_AtoB(a, b, triplet_block=triplet_block),
                    "yes", {"a": a, "b": b}))
    if "false" in a2b:
        a, b = a2b["false"]["a"], a2b["false"]["b"]
        out.append(("Flowchart-isFalse-AtoB",
                    flowlearn_isTrueFalse_AtoB(a, b, triplet_block=triplet_block),
                    "no", {"a": a, "b": b}))

    bet = sample.get("Arrow_betweenAB", {})
    if "true" in bet:
        a, b = bet["true"]["a"], bet["true"]["b"]
        out.append(("Flowchart-isTrue-betweenAB",
                    flowlearn_isTrueFalse_betweenAB(a, b, triplet_block=triplet_block),
                    "yes", {"a": a, "b": b}))
    if "false" in bet:
        a, b = bet["false"]["a"], bet["false"]["b"]
        out.append(("Flowchart-isFalse-betweenAB",
                    flowlearn_isTrueFalse_betweenAB(a, b, triplet_block=triplet_block),
                    "no", {"a": a, "b": b}))

    if "Num_Nodes" in sample:
        out.append(("Num_Nodes",
                    flowlearn_num_nodes_prompt(triplet_block=triplet_block),
                    str(sample["Num_Nodes"]), {}))
    if "Num_Arrows" in sample:
        out.append(("Num_Arrows",
                    flowlearn_num_arrows_prompt(triplet_block=triplet_block),
                    str(sample["Num_Arrows"]), {}))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    p.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    p.add_argument("--output-file", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--adapter-path", default="")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--triplets-dir", default="",
                   help="If set, inject LoRA-extracted triplets from "
                        "<triplets_dir>/<image_stem>/arrow_triplets.json into the prompt.")
    p.add_argument("--no-image", action="store_true",
                   help="Skip the image; query the model with text-only.")
    p.add_argument("--flowgen-mode", action="store_true",
                   help="Reproduce FlowGen's eval: only Arrow_betweenAB-true per image, "
                        "bare prompt with 'connectedto'.")
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
            existing = {}

    tasks = []
    skipped = 0
    plan: dict = {}  # (image_key, sub_key) -> {gt, extras, prompt}
    n_with_triplets = 0
    for k in keys:
        img_path = os.path.join(args.image_dir, k)  # k already includes ".jpeg"
        if not os.path.exists(img_path):
            print(f"[warn] missing image: {img_path}")
            continue
        image_stem = os.path.splitext(k)[0]
        triplets = _load_triplets(args.triplets_dir, image_stem) if args.triplets_dir else []
        triplet_block = format_triplet_block(triplets)
        if triplet_block:
            n_with_triplets += 1
        if args.flowgen_mode:
            # FlowGen evaluates only Arrow_betweenAB-true per image (always gt='yes').
            bet = gt[k].get("Arrow_betweenAB", {}).get("true")
            if not bet:
                continue
            a, b = bet["a"], bet["b"]
            prompt = (
                f"Determine whether the following description of the picture "
                f"is correct (just answer Yes or No) : '{a}' connectedto '{b}'"
            )
            questions = [("Flowchart-isTrue-betweenAB", prompt, "yes", {"a": a, "b": b})]
        else:
            questions = build_questions_for_image(gt[k], triplet_block=triplet_block)
        prior = existing.get(k, {})
        for sub_key, prompt, gt_val, extras in questions:
            plan[(k, sub_key)] = {"gt": gt_val, "extras": extras, "prompt": prompt}
            if sub_key in prior and prior[sub_key].get("predicted"):
                skipped += 1
                continue
            tasks.append({
                "image_path": img_path,
                "prompt": prompt,
                "out_key": (k, sub_key),
            })
    if args.triplets_dir:
        print(f"[plan] images with triplets loaded: {n_with_triplets}/{len(keys)}")

    print(f"[plan] images={len(keys)}  tasks={len(tasks)}  resume_skipped={skipped}")
    print(f"[plan] adapter={'(none)' if not args.adapter_path else args.adapter_path}")

    gpus = parse_gpus(args.gpus)
    results = run_multi_gpu(
        tasks,
        gpus=gpus,
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        system_prompt="" if args.flowgen_mode else SYSTEM_PROMPT,
        no_image=args.no_image,
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
            "ok": r["ok"],
        }

    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] saved → {args.output_file}  ({sum(len(v) for v in out.values())} predictions)")


if __name__ == "__main__":
    main()
