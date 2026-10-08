"""Run FlowVQA QA on a Qwen-VL model (with optional LoRA adapter).

For each test image (953 total), there are 8 questions (qa["1"]..qa["8"]).
We feed the whole image plus each question to the model, and store predictions.

Usage:
    python QA/infer_flowvqa.py \
        --output-file QA/results/flowvqa_qwen3_4b_arrow_lora.json \
        --base-model-path <MODEL_ROOT>/Qwen3-VL-4B-Instruct \
        --adapter-path .../qwen3_vl_4b_arrow_flowvqa_bs64/checkpoint-834 \
        --gpus 0,1,2,3 \
        --limit 200       # optional, for sanity check
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Make repo root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "QA"))

from prompts import flowvqa_qa_prompt, format_triplet_block, SYSTEM_PROMPT
from qa_runner import run_multi_gpu, parse_gpus


def _load_triplets(triplets_dir: str, image_stem: str):
    """Read arrow_triplets.json saved by lora/inference/get_triplets_sft_trace.py."""
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


DEFAULT_TEST_JSON = "<REPO_ROOT>/QA/textflow_subset_testjson/flowvqa_test.json"
DEFAULT_IMAGE_DIR = "<DATA_ROOT>/flowvqa/Data/A. Main Set Flowchart Images"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    p.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    p.add_argument("--output-file", required=True,
                   help="Output JSON path (single file with all predictions).")
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--adapter-path", default="",
                   help="LoRA adapter path. Empty = base model only (untrained baseline).")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=0,
                   help="Only process the first N images (0 = all).")
    p.add_argument("--resume", action="store_true",
                   help="If output file exists, skip image+qid pairs already present.")
    p.add_argument("--triplets-dir", default="",
                   help="If set, inject LoRA-extracted triplets from "
                        "<triplets_dir>/<image_stem>/arrow_triplets.json into the prompt.")
    p.add_argument("--no-image", action="store_true",
                   help="Skip the image; query the model with text-only "
                        "(prompt + triplets). Useful for ablations.")
    p.add_argument("--flowgen-mode", action="store_true",
                   help="Reproduce FlowGen's eval protocol: only qa['1'] per image, "
                        "bare `<image>{question}` prompt (no system prompt, no wrapper).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.test_json) as f:
        gt = json.load(f)

    keys = sorted(gt.keys())
    if args.limit > 0:
        keys = keys[: args.limit]

    # Resume support
    existing: dict = {}
    if args.resume and os.path.exists(args.output_file):
        try:
            with open(args.output_file) as f:
                existing = json.load(f)
            print(f"[resume] loaded {len(existing)} prior images from {args.output_file}")
        except Exception as e:
            print(f"[resume] failed to load existing file: {e}")
            existing = {}

    tasks = []
    skipped = 0
    n_with_triplets = 0
    for k in keys:
        sample = gt[k]
        img_path = os.path.join(args.image_dir, f"{k}.png")
        if not os.path.exists(img_path):
            print(f"[warn] missing image: {img_path}")
            continue
        qa_dict = sample.get("qa", {})
        prior = existing.get(k, {})
        triplets = _load_triplets(args.triplets_dir, k) if args.triplets_dir else []
        triplet_block = format_triplet_block(triplets)
        if triplet_block:
            n_with_triplets += 1
        if args.flowgen_mode:
            # FlowGen evaluates only qa['1'] per image, bare prompt.
            qid_list = ["1"]
        else:
            qid_list = list(qa_dict.keys())
        for qid in qid_list:
            if qid not in qa_dict:
                continue
            item = qa_dict[qid]
            if qid in prior and prior[qid].get("predicted"):
                skipped += 1
                continue
            if args.flowgen_mode:
                prompt = item["Q"]  # bare question, no wrapper
            else:
                prompt = flowvqa_qa_prompt(item["Q"], triplet_block=triplet_block)
            tasks.append({
                "image_path": img_path,
                "prompt": prompt,
                "out_key": (k, qid),
            })
    if args.triplets_dir:
        print(f"[plan] images with triplets loaded: {n_with_triplets}/{len(keys)}")

    print(f"[plan] images={len(keys)}  total tasks={len(tasks)}  resume_skipped={skipped}")
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

    # Merge into existing result dict
    out: dict = existing
    for r in results:
        k, qid = r["out_key"]
        if k not in out:
            out[k] = {}
        gt_item = gt[k]["qa"][qid]
        out[k][qid] = {
            "Q": gt_item["Q"],
            "A1": gt_item.get("A1", ""),
            "A2": gt_item.get("A2", ""),
            "A3": gt_item.get("A3", ""),
            "type": gt_item.get("type", ""),
            "predicted": r["predicted"],
            "ok": r["ok"],
        }

    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] saved → {args.output_file}  ({sum(len(v) for v in out.values())} predictions)")


if __name__ == "__main__":
    main()
