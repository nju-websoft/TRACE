"""
LOO triplet post-processing:

input:  output_lora_triplet_loo/<model>_triplet_loo_<target>_bs<N>_step<S>/predictions.json
        (dict: image_name → [{source, condition, end}, ...])

flow:
  For each image:
    1. find_image(stem, TEST_IMAGE_DIR[target]) locates the source image
    2. PaddleOCR builds the vocabulary
    3. correct_triplets fixes each triplet's source / end / condition

output: <loo_dir>_ocrfixed/predictions.json + ocr_correction_summary.json

usage:
  python -m postprocess.correct_loo_triplet \
      --loo-dir output_lora_triplet_loo/qwen3_vl_4b_triplet_loo_flowlearn_bs6_step818
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from postprocess.ocr_correction import (
    build_paddle_ocr,
    correct_triplets,
    extract_ocr_vocab,
    find_image,
)


FLOWQA_ROOT = "<REPO_ROOT>"

TEST_IMAGE_DIR: Dict[str, str] = {
    "fca":            f"{FLOWQA_ROOT}/Dataset/Block-Diagram-Datasets/Handwritten_block_diagrams/FC_A/test_img",
    "cbd":            f"{FLOWQA_ROOT}/Dataset/Block-Diagram-Datasets/Computerized_block_diagrams(CBD)/test_img",
    "fcb":            f"{FLOWQA_ROOT}/Dataset/Block-Diagram-Datasets/Handwritten_block_diagrams/FC_B/test",
    "flowlearn":      f"{FLOWQA_ROOT}/Dataset/flowlearn/mermaid_word/jpeg",
    "flowvqa":        f"{FLOWQA_ROOT}/Dataset/flowvqa/Data/A. Main Set Flowchart Images",
    "bpmn":           f"{FLOWQA_ROOT}/Dataset/bpmn/new_test_images",
    "flowgen_easy":   f"{FLOWQA_ROOT}/Dataset/flowgen/test_img_easy_orig",
    "flowgen_medium": f"{FLOWQA_ROOT}/Dataset/flowgen/test_img_medium_orig",
    "flowgen_hard":   f"{FLOWQA_ROOT}/Dataset/flowgen/test_img_hard_orig",
}

EVAL_PY = "<REPO_ROOT>/lora/eval/eval_E2E.py"
EVAL_DATASET: Dict[str, str] = {
    "fca": "fca", "cbd": "cbd", "fcb": "fcb",
    "flowlearn": "flowlearn", "flowvqa": "flowvqa", "bpmn": "bpmn",
    "flowgen_easy": "flowgen", "flowgen_medium": "flowgen", "flowgen_hard": "flowgen",
}

LOO_DIR_PATTERN = re.compile(
    r"^(?P<model>.+?)_triplet_loo_(?P<target>fca|cbd|fcb|flowlearn|flowvqa|bpmn|flowgen_easy|flowgen_medium|flowgen_hard)"
    r"_bs(?P<bs>\d+)_step(?P<step>\d+)$"
)


def parse_loo_dir(name: str) -> Optional[dict]:
    m = LOO_DIR_PATTERN.match(name)
    return m.groupdict() if m else None


def run_eval(target: str, prediction_file: Path) -> Optional[dict]:
    ds = EVAL_DATASET.get(target)
    if not ds or not Path(EVAL_PY).exists():
        return None
    cmd = [sys.executable, EVAL_PY, "--dataset", ds, "--prediction-file", str(prediction_file)]
    print(f"  $ {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=FLOWQA_ROOT, timeout=1200)
    except subprocess.TimeoutExpired:
        print("    eval timed out")
        return None
    # the eval script writes evaluation_results_*.json next to prediction_file
    parent = prediction_file.parent
    stem = prediction_file.stem
    for f in parent.glob(f"evaluation_results*{stem}*.json"):
        try:
            return json.load(open(f))
        except Exception:
            pass
    return None


def get_f1(eval_result):
    if not eval_result:
        return None
    if "f1_score" in eval_result:
        return float(eval_result["f1_score"])
    if "relaxed_metrics" in eval_result:
        return float(eval_result["relaxed_metrics"]["f1"])
    return None


def process(loo_dir: Path, max_dist: int = 2, skip_condition: bool = False,
            do_eval: bool = True, target_override: Optional[str] = None):
    info = parse_loo_dir(loo_dir.name)
    if not info:
        if not target_override:
            raise ValueError(f"cannot parse {loo_dir.name} (use --target)")
        info = {"model": "?", "target": target_override, "bs": "?", "step": "?"}
    target = target_override or info["target"]
    image_dir = TEST_IMAGE_DIR[target]
    pred_file = loo_dir / "predictions.json"
    if not pred_file.exists():
        raise FileNotFoundError(f"missing {pred_file}")

    out_dir = loo_dir.parent / f"{loo_dir.name}_ocrfixed"
    out_dir.mkdir(exist_ok=True)
    out_pred = out_dir / "predictions.json"
    summary_path = out_dir / "ocr_correction_summary.json"

    print(f"loo_dir   : {loo_dir}")
    print(f"target    : {target}")
    print(f"image_dir : {image_dir}")
    print(f"out_dir   : {out_dir}")
    print(f"max_dist  : {max_dist}")
    print()

    with open(pred_file) as f:
        preds = json.load(f)

    print(f"loading PaddleOCR (en_mobile)...")
    ocr = build_paddle_ocr("en_mobile")

    n_imgs = len(preds)
    n_corrected = n_no_image = n_no_vocab = 0
    total_triplets = 0
    total_changed = 0

    new_preds = {}
    t0 = time.time()
    for i, (img_name, triplets) in enumerate(preds.items(), 1):
        stem = Path(img_name).stem
        img_path = find_image(stem, image_dir)
        if img_path is None:
            new_preds[img_name] = triplets
            n_no_image += 1
            continue
        try:
            vocab = extract_ocr_vocab(str(img_path), ocr)
        except Exception as e:
            print(f"  [{i}/{n_imgs}] OCR fail {stem}: {e}")
            new_preds[img_name] = triplets
            continue
        if not vocab:
            new_preds[img_name] = triplets
            n_no_vocab += 1
            continue
        fixed, n_changed = correct_triplets(
            triplets, vocab, max_dist=max_dist, skip_condition=skip_condition,
        )
        new_preds[img_name] = fixed
        total_triplets += len(triplets)
        total_changed += n_changed
        if n_changed > 0:
            n_corrected += 1
        if i % 50 == 0:
            rate = i / (time.time() - t0)
            eta = (n_imgs - i) / rate
            print(f"  [{i}/{n_imgs}] {n_corrected} corrected, {total_changed} fields changed  "
                  f"({rate:.2f} img/s, ETA {eta/60:.1f}min)")

    with open(out_pred, "w", encoding="utf-8") as f:
        json.dump(new_preds, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_pred}")

    eval_before = eval_after = None
    if do_eval:
        print("\nbefore eval (original)...")
        eval_before = run_eval(target, pred_file)
        print("after eval (OCR-corrected)...")
        eval_after = run_eval(target, out_pred)

    summary = {
        "loo_dir": str(loo_dir),
        "out_dir": str(out_dir),
        "target": target,
        "model": info["model"],
        "max_dist": max_dist,
        "correction_stats": {
            "images_total": n_imgs,
            "images_corrected": n_corrected,
            "images_no_image": n_no_image,
            "images_no_vocab": n_no_vocab,
            "triplets_total": total_triplets,
            "fields_changed": total_changed,
        },
        "f1_before": get_f1(eval_before),
        "f1_after":  get_f1(eval_after),
    }
    if summary["f1_before"] is not None and summary["f1_after"] is not None:
        summary["f1_delta"] = summary["f1_after"] - summary["f1_before"]

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loo-dir", required=True)
    p.add_argument("--max-dist", type=int, default=2)
    p.add_argument("--skip-condition", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--target", default=None)
    a = p.parse_args()
    process(Path(a.loo_dir), max_dist=a.max_dist, skip_condition=a.skip_condition,
            do_eval=not a.no_eval, target_override=a.target)


if __name__ == "__main__":
    main()
