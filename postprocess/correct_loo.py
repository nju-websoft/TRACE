#!/usr/bin/env python3
"""
LOO post-processing driver:
  1. Scan one LOO output dir <model>_arrow_loo_<target>_bs<N>_step<S>_<arrow_src>/
  2. Parse target -> know which source-image dir and eval script to use
  3. Per image: PaddleOCR vocab -> Levenshtein-correct source/end/condition
  4. Write corrected results to the parallel dir <orig>_ocrfixed/
  5. Run the matching eval on the old and new dirs and print a comparison

Usage:
  python -m postprocess.correct_loo --loo-dir <OUTPUT_ROOT>/qwen3_vl_4b_arrow_loo_fca_bs48_step3163_sam
  python -m postprocess.correct_loo --loo-dir <dir> --max-dist 3 --skip-condition --no-eval
  python -m postprocess.correct_loo --all                       # process every LOO dir under output_lora_arrow_loo
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from postprocess.ocr_correction import (
    build_paddle_ocr,
    build_paddle_ocrs,
    correct_triplets,
    extract_ocr_vocab,
    extract_ocr_vocab_union,
    find_image,
)


# ---------------------------------------------------------------------------
# Dataset -> test-image dir, eval script, GT root (same config as the arrow eval)
# ---------------------------------------------------------------------------
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

EVAL_PY = "<REPO_ROOT>/lora/eval/eval_TRACE.py"
EVAL_DATASET: Dict[str, str] = {
    "fca": "fca", "cbd": "cbd", "fcb": "fcb",
    "flowlearn": "flowlearn", "flowvqa": "flowvqa", "bpmn": "bpmn",
    "flowgen_easy": "flowgen", "flowgen_medium": "flowgen", "flowgen_hard": "flowgen",
}

EVAL_GT_ROOT: Dict[str, str] = {
    "flowgen_easy":   f"{FLOWQA_ROOT}/Dataset/data_4_training/flowgen_test_easy",
    "flowgen_medium": f"{FLOWQA_ROOT}/Dataset/data_4_training/flowgen_test_medium",
    "flowgen_hard":   f"{FLOWQA_ROOT}/Dataset/data_4_training/flowgen_test_hard",
}

# Dir-name pattern: <model>_arrow_loo_<target>_bs<N>_step<S>_<arrow_src>
LOO_DIR_PATTERN = re.compile(
    r"^(?P<model>.+?)_arrow_loo_(?P<target>fca|cbd|fcb|flowlearn|flowvqa|bpmn|flowgen_easy|flowgen_medium|flowgen_hard)"
    r"_bs(?P<bs>\d+)_step(?P<step>\d+)_(?P<src>.+)$"
)


def parse_loo_dir(name: str) -> Optional[dict]:
    m = LOO_DIR_PATTERN.match(name)
    return m.groupdict() if m else None


def fake_info(name: str, target: str) -> dict:
    """When the dir name does not match the LOO rule but the user gave --target, build a minimal info."""
    return {"model": name, "target": target, "bs": "0", "step": "0", "src": "manual"}


# ---------------------------------------------------------------------------
# Core: OCR-correct one LOO directory
# ---------------------------------------------------------------------------
def correct_loo_dir(
    loo_dir: Path,
    image_dir: Path,
    out_dir: Path,
    ocr_model,
    max_dist: int = 2,
    skip_condition: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
    min_ocr_score: float = 0.0,
    min_value_len: int = 0,
    garbled_only: bool = False,
) -> dict:
    """
    Scan loo_dir/<image_stem>/arrow_triplets.json,
    After OCR correction, write to out_dir/<image_stem>/arrow_triplets.json.
    If num_shards>1, only process subdirs where idx % num_shards == shard_id.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    subs = sorted([p for p in loo_dir.iterdir() if p.is_dir() and (p / "arrow_triplets.json").exists()])
    if num_shards > 1:
        subs = [s for i, s in enumerate(subs) if i % num_shards == shard_id]
        print(f"  [shard {shard_id}/{num_shards}] assigned {len(subs)} images")
    n_total_triplets = 0
    n_changed_fields = 0
    n_imgs_corrected = 0
    n_imgs_no_image = 0
    n_imgs_no_vocab = 0

    for i, sub in enumerate(subs, 1):
        stem = sub.name
        triplets_path = sub / "arrow_triplets.json"
        with triplets_path.open("r", encoding="utf-8") as f:
            triplets = json.load(f)
        n_total_triplets += len(triplets)

        target_sub = out_dir / stem
        target_sub.mkdir(exist_ok=True)
        out_path = target_sub / "arrow_triplets.json"

        img_path = find_image(stem, str(image_dir))
        if img_path is None:
            n_imgs_no_image += 1
            shutil.copyfile(triplets_path, out_path)  # passthrough
            print(f"  [{i}/{len(subs)}] {stem}: no source image, copied as-is")
            continue

        if isinstance(ocr_model, list):
            vocab = extract_ocr_vocab_union(str(img_path), ocr_model, min_score=min_ocr_score)
        else:
            vocab = extract_ocr_vocab(str(img_path), ocr_model, min_score=min_ocr_score)
        if not vocab:
            n_imgs_no_vocab += 1
            shutil.copyfile(triplets_path, out_path)
            print(f"  [{i}/{len(subs)}] {stem}: OCR found no text, copied as-is")
            continue

        corrected, changed = correct_triplets(
            triplets, vocab, max_dist=max_dist, skip_condition=skip_condition,
            min_len=min_value_len, garbled_only=garbled_only,
        )
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(corrected, f, ensure_ascii=False, indent=2)

        n_changed_fields += changed
        if changed > 0:
            n_imgs_corrected += 1
            print(f"  [{i}/{len(subs)}] {stem}: changed {changed} fields ({len(vocab)} words)")
        else:
            print(f"  [{i}/{len(subs)}] {stem}: no change ({len(vocab)} words)")

    return {
        "images_total": len(subs),
        "images_corrected": n_imgs_corrected,
        "images_no_image": n_imgs_no_image,
        "images_no_vocab": n_imgs_no_vocab,
        "triplets_total": n_total_triplets,
        "fields_changed": n_changed_fields,
    }


# ---------------------------------------------------------------------------
# Scoring: call the matching eval script
# ---------------------------------------------------------------------------
def run_eval(target: str, output_dir: Path) -> Optional[dict]:
    ds = EVAL_DATASET.get(target)
    if not ds or not Path(EVAL_PY).exists():
        print(f"  [eval] eval script/dataset mapping not found: {EVAL_PY} (target={target})")
        return None
    cmd = [sys.executable, EVAL_PY, "--dataset", ds, "--output-dir", str(output_dir)]
    if target in EVAL_GT_ROOT:
        cmd += ["--gt-root", EVAL_GT_ROOT[target]]
    print(f"  [eval] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=FLOWQA_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("  [eval] failed:")
        print("    stderr:", proc.stderr[-500:])
        return None
    ej = output_dir / "evaluation_results.json"
    if not ej.exists():
        print(f"  [eval] score output not produced: {ej}")
        return None
    with ej.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_f1(eval_dict: Optional[dict]) -> Optional[float]:
    """Each dataset's evaluation_results.json differs; uniformly read the standard F1."""
    if not eval_dict:
        return None
    for path in [
        ("no_compensation", "standard_metrics", "f1"),
        ("standard_metrics", "f1"),
        ("f1_score",),
        ("relaxed_f1",),
        ("f1",),
    ]:
        cur = eval_dict
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            return float(cur)
    return None


# ---------------------------------------------------------------------------
# Single LOO dir, end-to-end
# ---------------------------------------------------------------------------
def process_one(
    loo_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    image_dir_override: Optional[Path] = None,
    max_dist: int = 2,
    skip_condition: bool = False,
    do_eval: bool = True,
    ocr_model=None,
    shard_id: int = 0,
    num_shards: int = 1,
    min_ocr_score: float = 0.0,
    min_value_len: int = 0,
    garbled_only: bool = False,
    target_override: Optional[str] = None,
) -> dict:
    info = parse_loo_dir(loo_dir.name)
    if not info:
        if not target_override:
            raise ValueError(f"cannot parse LOO dir name: {loo_dir.name} (specify with --target)")
        info = fake_info(loo_dir.name, target_override)
    target = target_override or info["target"]

    image_dir = image_dir_override or Path(TEST_IMAGE_DIR[target])
    if not image_dir.exists():
        raise FileNotFoundError(f"test-image dir not found: {image_dir}")

    out_dir = out_dir or loo_dir.parent / f"{loo_dir.name}_ocrfixed"
    print(f"\n=== LOO correction ===")
    print(f"  loo_dir : {loo_dir}")
    print(f"  target  : {target}")
    print(f"  images  : {image_dir}")
    print(f"  out_dir : {out_dir}")
    print(f"  max_dist: {max_dist}, skip_cond: {skip_condition}")

    if ocr_model is None:
        print("  [init] loading PaddleOCR ...")
        ocr_model = build_paddle_ocr()

    correction_stats = correct_loo_dir(
        loo_dir, image_dir, out_dir, ocr_model,
        max_dist=max_dist, skip_condition=skip_condition,
        shard_id=shard_id, num_shards=num_shards,
        min_ocr_score=min_ocr_score,
        min_value_len=min_value_len,
        garbled_only=garbled_only,
    )

    eval_before = eval_after = None
    f1_before = f1_after = None
    if do_eval and num_shards > 1:
        print(f"  [eval] shard mode skips scoring (run it after merging all shards)")
        do_eval = False
    if do_eval:
        print("\n  [eval] score before:")
        eval_before = run_eval(target, loo_dir)
        f1_before = extract_f1(eval_before)
        print("  [eval] score after:")
        eval_after = run_eval(target, out_dir)
        f1_after = extract_f1(eval_after)

    summary = {
        "loo_dir": str(loo_dir),
        "out_dir": str(out_dir),
        "target": target,
        "model": info["model"],
        "bs": int(info["bs"]),
        "step": int(info["step"]),
        "arrow_src": info["src"],
        "max_dist": max_dist,
        "skip_condition": skip_condition,
        "correction_stats": correction_stats,
        "f1_before": f1_before,
        "f1_after": f1_after,
        "f1_delta": (f1_after - f1_before) if (f1_before is not None and f1_after is not None) else None,
    }

    if num_shards > 1:
        summary_path = out_dir / f"ocr_correction_summary_shard{shard_id}of{num_shards}.json"
    else:
        summary_path = out_dir / "ocr_correction_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(f"  fields changed: {correction_stats['fields_changed']} / {correction_stats['triplets_total']*3} field slots")
    if f1_before is not None and f1_after is not None:
        sign = "+" if f1_after >= f1_before else ""
        print(f"  F1 before -> after: {f1_before:.4f} -> {f1_after:.4f}  ({sign}{f1_after - f1_before:+.4f})")
    print(f"  summary: {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# Finalize: merge shard summaries + run eval (no OCR / GPU needed)
# ---------------------------------------------------------------------------
def finalize_dir(loo_dir: Path, out_dir: Optional[Path] = None, do_eval: bool = True,
                 target_override: Optional[str] = None) -> dict:
    info = parse_loo_dir(loo_dir.name)
    if not info:
        if not target_override:
            raise ValueError(f"cannot parse LOO dir name: {loo_dir.name} (specify with --target)")
        info = fake_info(loo_dir.name, target_override)
    target = target_override or info["target"]
    out_dir = out_dir or loo_dir.parent / f"{loo_dir.name}_ocrfixed"
    if not out_dir.exists():
        raise FileNotFoundError(f"output dir not found: {out_dir}")

    print(f"\n=== Finalize ===")
    print(f"  loo_dir : {loo_dir}")
    print(f"  out_dir : {out_dir}")

    # merge shard summaries
    shard_files = sorted(out_dir.glob("ocr_correction_summary_shard*.json"))
    if not shard_files:
        print("  [warn] no shard summary found; scoring only")
        merged_stats = {"images_total": 0, "images_corrected": 0, "images_no_image": 0,
                        "images_no_vocab": 0, "triplets_total": 0, "fields_changed": 0}
    else:
        merged_stats = {"images_total": 0, "images_corrected": 0, "images_no_image": 0,
                        "images_no_vocab": 0, "triplets_total": 0, "fields_changed": 0}
        for sf in shard_files:
            with sf.open("r", encoding="utf-8") as f:
                s = json.load(f)
            cs = s.get("correction_stats", {})
            for k in merged_stats:
                merged_stats[k] += cs.get(k, 0)
        print(f"  merged {len(shard_files)} shards: {merged_stats}")

    f1_before = f1_after = None
    if do_eval:
        print("  [eval] score before:")
        f1_before = extract_f1(run_eval(target, loo_dir))
        print("  [eval] score after:")
        f1_after = extract_f1(run_eval(target, out_dir))

    summary = {
        "loo_dir": str(loo_dir),
        "out_dir": str(out_dir),
        "target": target,
        "model": info["model"],
        "bs": int(info["bs"]),
        "step": int(info["step"]),
        "arrow_src": info["src"],
        "correction_stats": merged_stats,
        "num_shards_merged": len(shard_files),
        "f1_before": f1_before,
        "f1_after": f1_after,
        "f1_delta": (f1_after - f1_before) if (f1_before is not None and f1_after is not None) else None,
    }
    summary_path = out_dir / "ocr_correction_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(f"  fields changed: {merged_stats['fields_changed']} / {merged_stats['triplets_total']*3} field slots")
    if f1_before is not None and f1_after is not None:
        print(f"  F1 before -> after: {f1_before:.4f} -> {f1_after:.4f}  ({f1_after - f1_before:+.4f})")
    print(f"  summary: {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--loo-dir", type=str, help="A single LOO directory")
    g.add_argument("--all", action="store_true",
                   help=f"Process every dir under {FLOWQA_ROOT}/output_lora_arrow_loo/ matching the LOO naming rule")
    ap.add_argument("--out-dir", type=str, help="Output dir (default: <loo_dir>_ocrfixed)")
    ap.add_argument("--image-dir", type=str, help="Override the source-image dir (default: auto-selected by target)")
    ap.add_argument("--max-dist", type=int, default=2, help="Levenshtein distance threshold (default 2)")
    ap.add_argument("--skip-condition", action="store_true", help="Do not correct the condition field")
    ap.add_argument("--no-eval", action="store_true", help="Only correct; do not re-score")
    ap.add_argument("--root", type=str, default=f"{FLOWQA_ROOT}/output_lora_arrow_loo",
                    help="Root dir to scan in --all mode")
    ap.add_argument("--shard-id", type=int, default=0, help="This process's shard id (0..num_shards-1)")
    ap.add_argument("--num-shards", type=int, default=1, help="Total number of shards (set >1 for multi-GPU parallelism)")
    ap.add_argument("--ocr-presets", type=str, default="en_mobile",
                    help="Comma-separated PaddleOCR presets; multiple presets union their vocab. Options: en_mobile,ch_server")
    ap.add_argument("--min-ocr-score", type=float, default=0.0,
                    help="OCR token confidence threshold; tokens below it are dropped")
    ap.add_argument("--min-value-len", type=int, default=0,
                    help="Do not correct fields shorter than this (avoids mangling Y/N, i++, etc.)")
    ap.add_argument("--garbled-only", action="store_true",
                    help="Only replace when the original value does not look like an English word (via wordfreq). Recommended for FCA/FCB; disable for FlowLearn.")
    ap.add_argument("--finalize-only", action="store_true",
                    help="Skip OCR; only merge shard summaries + run scoring (call after all GPUs finish)")
    ap.add_argument("--target", type=str, default=None,
                    help="Explicit dataset (fca/fcb/flowlearn/...), for when the dir name does not follow the LOO naming")
    args = ap.parse_args()
    if not (0 <= args.shard_id < args.num_shards):
        sys.exit(f"--shard-id must be in [0, {args.num_shards})")

    if args.finalize_only:
        if not args.loo_dir:
            sys.exit("--finalize-only must be used with --loo-dir")
        finalize_dir(
            loo_dir=Path(args.loo_dir),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            do_eval=not args.no_eval,
            target_override=args.target,
        )
        return

    targets: List[Path] = []
    if args.loo_dir:
        p = Path(args.loo_dir)
        if not p.is_dir():
            sys.exit(f"dir not found: {p}")
        targets = [p]
        if args.target is None and parse_loo_dir(p.name) is None:
            sys.exit(f"cannot parse LOO dir name '{p.name}'; specify the dataset with --target")
    else:
        root = Path(args.root)
        for d in sorted(root.iterdir()):
            if d.is_dir() and parse_loo_dir(d.name) and not d.name.endswith("_ocrfixed"):
                targets.append(d)
        if not targets:
            sys.exit(f"no LOO dir found under {root}")
        print(f"will process {len(targets)} LOO dirs:")
        for t in targets:
            print(f"  - {t.name}")

    presets = [p.strip() for p in args.ocr_presets.split(",") if p.strip()]
    print(f"\n[init] loading PaddleOCR presets: {presets} ...")
    ocr_model = build_paddle_ocrs(presets) if len(presets) > 1 else build_paddle_ocr(presets[0])

    all_summaries = []
    for loo_dir in targets:
        try:
            s = process_one(
                loo_dir,
                out_dir=Path(args.out_dir) if args.out_dir else None,
                image_dir_override=Path(args.image_dir) if args.image_dir else None,
                max_dist=args.max_dist,
                skip_condition=args.skip_condition,
                do_eval=not args.no_eval,
                ocr_model=ocr_model,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                min_ocr_score=args.min_ocr_score,
                min_value_len=args.min_value_len,
                garbled_only=args.garbled_only,
                target_override=args.target,
            )
            all_summaries.append(s)
        except Exception as e:
            print(f"\n[ERROR] failed processing {loo_dir.name}: {e}")

    if len(all_summaries) > 1:
        print("\n" + "=" * 80)
        print("OCR correction + re-eval summary")
        print("=" * 80)
        print(f"  {'target':<16} {'model':<16} {'F1 before':>10} {'F1 after':>10} {'Δ':>8}")
        for s in all_summaries:
            fb = f"{s['f1_before']:.4f}" if s["f1_before"] is not None else "N/A"
            fa = f"{s['f1_after']:.4f}" if s["f1_after"] is not None else "N/A"
            d = f"{s['f1_delta']:+.4f}" if s["f1_delta"] is not None else "N/A"
            print(f"  {s['target']:<16} {s['model']:<16} {fb:>10} {fa:>10} {d:>8}")


if __name__ == "__main__":
    main()
