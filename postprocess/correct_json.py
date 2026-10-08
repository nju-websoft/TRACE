#!/usr/bin/env python3
"""Single-JSON-file OCR post-processing (output_lora_triplet/<model>/<dataset>_<bsN>_best.json format).

Input: {<image_stem.ext>: [{source, end, condition, ...}, ...], ...}
Output: same-structure JSON, with source/end corrected via OCR-vocab edit distance

Supports shard parallelism + finalize merge + running the eval for comparison.

Usage:
  # single process
  python -m postprocess.correct_json \\
      --prediction-file output_lora_triplet/qwen3_vl_4b/flowlearn_bs16_best.json \\
      --target flowlearn

  # multi-GPU sharding
  CUDA_VISIBLE_DEVICES=0 python -m postprocess.correct_json \\
      --prediction-file ... --target flowlearn --shard-id 0 --num-shards 2 --no-eval &
  CUDA_VISIBLE_DEVICES=1 python -m postprocess.correct_json \\
      --prediction-file ... --target flowlearn --shard-id 1 --num-shards 2 --no-eval &
  wait
  python -m postprocess.correct_json --prediction-file ... --target flowlearn --finalize-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from postprocess.ocr_correction import (
    build_paddle_ocr,
    build_paddle_ocrs,
    correct_triplets,
    extract_ocr_vocab,
    extract_ocr_vocab_union,
    find_image,
)
from postprocess.correct_loo import (
    TEST_IMAGE_DIR,
    EVAL_GT_ROOT,
    extract_f1,
    FLOWQA_ROOT,
)

# Single-JSON eval script (accepts --prediction-file)
EVAL_PY = "<REPO_ROOT>/lora/eval/eval_E2E.py"
EVAL_DATASET: Dict[str, str] = {
    "fca": "fca", "cbd": "cbd", "fcb": "fcb",
    "flowlearn": "flowlearn", "flowvqa": "flowvqa", "bpmn": "bpmn",
    "flowgen_easy": "flowgen", "flowgen_medium": "flowgen", "flowgen_hard": "flowgen",
}


def correct_one_shard(
    pred: Dict[str, list],
    image_dir: Path,
    ocr_model,
    *,
    max_dist: int,
    skip_condition: bool,
    min_ocr_score: float,
    min_value_len: int,
    garbled_only: bool,
    shard_id: int,
    num_shards: int,
) -> tuple[Dict[str, list], dict]:
    """Process one shard of pred; return (corrected_pred_subset, stats)."""
    keys = sorted(pred.keys())
    if num_shards > 1:
        keys = [k for i, k in enumerate(keys) if i % num_shards == shard_id]
        print(f"  [shard {shard_id}/{num_shards}] assigned {len(keys)} images")

    out: Dict[str, list] = {}
    n_total_triplets = 0
    n_changed = 0
    n_corrected_imgs = 0
    n_no_image = 0
    n_no_vocab = 0

    for i, k in enumerate(keys, 1):
        triplets = pred[k]
        n_total_triplets += len(triplets)
        # k looks like '8394.jpeg'; find_image accepts a bare stem or one with a suffix
        img_path = find_image(k, str(image_dir))
        if img_path is None:
            img_path = find_image(Path(k).stem, str(image_dir))
        if img_path is None:
            n_no_image += 1
            out[k] = triplets
            print(f"  [{i}/{len(keys)}] {k}: no source image, kept as-is")
            continue
        if isinstance(ocr_model, list):
            vocab = extract_ocr_vocab_union(str(img_path), ocr_model, min_score=min_ocr_score)
        else:
            vocab = extract_ocr_vocab(str(img_path), ocr_model, min_score=min_ocr_score)
        if not vocab:
            n_no_vocab += 1
            out[k] = triplets
            print(f"  [{i}/{len(keys)}] {k}: OCR found no text, kept as-is")
            continue
        corrected, changed = correct_triplets(
            triplets, vocab, max_dist=max_dist, skip_condition=skip_condition,
            min_len=min_value_len, garbled_only=garbled_only,
        )
        out[k] = corrected
        n_changed += changed
        if changed > 0:
            n_corrected_imgs += 1
            print(f"  [{i}/{len(keys)}] {k}: changed {changed} fields ({len(vocab)} words)")
    stats = {
        "images_total": len(keys),
        "images_corrected": n_corrected_imgs,
        "images_no_image": n_no_image,
        "images_no_vocab": n_no_vocab,
        "triplets_total": n_total_triplets,
        "fields_changed": n_changed,
    }
    return out, stats


def run_eval(target: str, prediction_file: Path) -> Optional[dict]:
    """Call the eval script with --prediction-file and read its output JSON."""
    ds = EVAL_DATASET.get(target)
    if not ds or not Path(EVAL_PY).exists():
        print(f"  [eval] eval script/dataset mapping not found: {EVAL_PY} (target={target})")
        return None
    out_eval = prediction_file.parent / f"evaluation_results_{prediction_file.stem}.json"
    cmd = [sys.executable, EVAL_PY, "--dataset", ds,
           "--prediction-file", str(prediction_file),
           "--output-file", str(out_eval)]
    if target in EVAL_GT_ROOT:
        cmd += ["--gt-root", EVAL_GT_ROOT[target]]
    print(f"  [eval] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=FLOWQA_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("  [eval] failed:")
        print("    stderr:", proc.stderr[-500:])
        return None
    if not out_eval.exists():
        print(f"  [eval] score output not produced: {out_eval}")
        return None
    with out_eval.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--prediction-file", type=str, required=True,
                    help="Source single JSON file ({stem.ext: [triplets]})")
    ap.add_argument("--target", type=str, required=True, choices=list(TEST_IMAGE_DIR),
                    help="Dataset name (determines source-image dir and eval script)")
    ap.add_argument("--image-dir", type=str, default=None,
                    help="Override the test-image dir (default: auto-selected by target)")
    ap.add_argument("--out-file", type=str, default=None,
                    help="Corrected-JSON path (default <pred>_ocrfixed.json)")
    ap.add_argument("--max-dist", type=int, default=2)
    ap.add_argument("--skip-condition", action="store_true")
    ap.add_argument("--ocr-presets", type=str, default="en_mobile",
                    help="Comma-separated PaddleOCR presets")
    ap.add_argument("--min-ocr-score", type=float, default=0.0)
    ap.add_argument("--min-value-len", type=int, default=0)
    ap.add_argument("--garbled-only", action="store_true")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--finalize-only", action="store_true",
                    help="Skip OCR; only merge shard partial JSONs + score")
    args = ap.parse_args()

    pred_path = Path(args.prediction_file)
    if not pred_path.exists():
        sys.exit(f"prediction file not found: {pred_path}")
    out_path = Path(args.out_file) if args.out_file else pred_path.with_name(f"{pred_path.stem}_ocrfixed.json")

    # finalize: merge all shard partial files
    if args.finalize_only:
        merged: Dict[str, list] = {}
        merged_stats = {"images_total": 0, "images_corrected": 0, "images_no_image": 0,
                        "images_no_vocab": 0, "triplets_total": 0, "fields_changed": 0}
        partials = sorted(out_path.parent.glob(f"{out_path.stem}_shard*of*.json"))
        if not partials:
            sys.exit(f"no shard partials found: {out_path.parent}/{out_path.stem}_shard*of*.json")
        for p in partials:
            d = json.loads(p.read_text())
            merged.update(d.get("predictions", {}))
            for k in merged_stats:
                merged_stats[k] += d.get("stats", {}).get(k, 0)
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        print(f"[finalize] merged {len(partials)} shards -> {out_path}  ({len(merged)} keys, {merged_stats})")

        f1_before = f1_after = None
        eval_before = eval_after = None
        if not args.no_eval:
            print("[eval] before:")
            eval_before = run_eval(args.target, pred_path)
            f1_before = extract_f1(eval_before)
            print("[eval] after:")
            eval_after = run_eval(args.target, out_path)
            f1_after = extract_f1(eval_after)

        def rf1(ed):
            try:
                return ed["relaxed_metrics"]["f1"]
            except Exception:
                return None

        summary = {
            "prediction_file": str(pred_path),
            "out_file": str(out_path),
            "target": args.target,
            "correction_stats": merged_stats,
            "num_shards_merged": len(partials),
            "f1_before": f1_before,
            "f1_after": f1_after,
            "f1_delta": (f1_after - f1_before) if (f1_before is not None and f1_after is not None) else None,
            "rf1_before": rf1(eval_before) if eval_before else None,
            "rf1_after": rf1(eval_after) if eval_after else None,
        }
        summary_path = out_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print("\n=== Done ===")
        if f1_before is not None and f1_after is not None:
            print(f"  F1: {f1_before:.4f} → {f1_after:.4f}  ({f1_after - f1_before:+.4f})")
        if summary["rf1_before"] is not None and summary["rf1_after"] is not None:
            print(f"  rF1: {summary['rf1_before']:.4f} → {summary['rf1_after']:.4f}  ({summary['rf1_after'] - summary['rf1_before']:+.4f})")
        print(f"  summary: {summary_path}")
        return

    # normal path: OCR correction
    image_dir = Path(args.image_dir) if args.image_dir else Path(TEST_IMAGE_DIR[args.target])
    if not image_dir.exists():
        sys.exit(f"image dir not found: {image_dir}")
    pred = json.loads(pred_path.read_text())
    print(f"[init] loaded {len(pred)} predictions")

    presets = [p.strip() for p in args.ocr_presets.split(",") if p.strip()]
    print(f"[init] loading PaddleOCR presets: {presets}")
    ocr_model = build_paddle_ocrs(presets) if len(presets) > 1 else build_paddle_ocr(presets[0])

    corrected, stats = correct_one_shard(
        pred, image_dir, ocr_model,
        max_dist=args.max_dist,
        skip_condition=args.skip_condition,
        min_ocr_score=args.min_ocr_score,
        min_value_len=args.min_value_len,
        garbled_only=args.garbled_only,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    if args.num_shards > 1:
        partial = out_path.parent / f"{out_path.stem}_shard{args.shard_id}of{args.num_shards}.json"
        partial.write_text(json.dumps({"predictions": corrected, "stats": stats}, ensure_ascii=False))
        print(f"[shard {args.shard_id}/{args.num_shards}] partial → {partial}  ({stats})")
        return

    # single-process mode
    out_path.write_text(json.dumps(corrected, ensure_ascii=False, indent=2))
    print(f"[done] wrote {out_path}  ({stats})")

    if not args.no_eval:
        print("\n[eval] before:")
        eval_before = run_eval(args.target, pred_path)
        f1_before = extract_f1(eval_before)
        print("[eval] after:")
        eval_after = run_eval(args.target, out_path)
        f1_after = extract_f1(eval_after)
        if f1_before is not None and f1_after is not None:
            print(f"\n  F1: {f1_before:.4f} → {f1_after:.4f}  ({f1_after - f1_before:+.4f})")
        try:
            rb = eval_before["relaxed_metrics"]["f1"]
            ra = eval_after["relaxed_metrics"]["f1"]
            print(f"  rF1: {rb:.4f} → {ra:.4f}  ({ra - rb:+.4f})")
        except Exception:
            pass


if __name__ == "__main__":
    main()
