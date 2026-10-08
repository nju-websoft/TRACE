#!/usr/bin/env python3
"""
[OPTIONAL] Merge overlapping arrowheads in the intermediate data.

Some flowcharts have several arrowheads that almost coincide. If you want to train
the model to read such a cluster as a single bounding box (and answer with all the
connections at once), run this optional step. It:

  1. groups arrows whose arrowhead bboxes overlap (IoU >= threshold),
  2. merges each group into one arrow (union bbox; every field ' || '-joined),
  3. redraws ONE blue box per merged group on the clean whole image
     (saved to <sample>/annotated_images_merged/),
  4. writes "<name>_merged.json" next to the original "<name>.json".

`format_data.py` prefers "*_merged.json" when present, so the merged answers feed
straight into training. This step is entirely optional — skip it and training just
uses the per-arrow data as usual.

Works on any dataset's intermediate directory (the per-sample folders produced by the
gen_data_from_*.py scripts). The whole image (needed to redraw) is resolved from the
JSON's own 'image_path', else <image-dir>/<folder>.(png|jpeg), else a same-named image
inside the sample folder.

Usage:
    python gen_data/merge_arrows.py --data-dir <intermediate_dir> [--image-dir <whole_image_dir>]
                                    [--iou-threshold 0.6]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2

BOX_LINE_WIDTH = 3


# ── Field / bbox helpers (tolerate str-or-list fields and either bbox key) ──────

def _bbox_of(arrow: dict):
    return arrow.get("arrowhead_bbox") or arrow.get("matched_arrowhead_bbox")


def _field(arrow: dict, key: str) -> str:
    """Return a field as a ' || '-joined string (handles str or list)."""
    v = arrow.get(key, "")
    if isinstance(v, list):
        return " || ".join(str(x).strip() for x in v if x is not None and str(x).strip())
    return str(v).strip() if v is not None else ""


# ── IoU + union-find merge ──────────────────────────────────────────────────────

def calculate_iou(b1: List[float], b2: List[float]) -> float:
    x1_1, y1_1, x2_1, y2_1 = b1
    x1_2, y1_2, x2_2, y2_2 = b2
    ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
    ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def merge_arrows(arrows: List[Dict], iou_threshold: float = 0.6) -> List[Dict]:
    """Union-find merge arrows whose arrowhead bboxes overlap; ' || '-join the fields.
    Location fields are only emitted if the source arrows carry them (bpmn/flowgen)."""
    if not arrows:
        return []

    n = len(arrows)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    def _answer_key(a):
        return (_field(a, "source_node"), _field(a, "target_node"), _field(a, "condition"))

    for i in range(n):
        bi = _bbox_of(arrows[i])
        for j in range(i + 1, n):
            bj = _bbox_of(arrows[j])
            if bi and bj and calculate_iou(bi, bj) >= iou_threshold \
                    and _answer_key(arrows[i]) != _answer_key(arrows[j]):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    has_loc = any("source_location" in a or "target_location" in a for a in arrows)
    bbox_key = "arrowhead_bbox" if "arrowhead_bbox" in arrows[0] else "matched_arrowhead_bbox"

    merged = []
    for indices in groups.values():
        group = [arrows[i] for i in indices]
        boxes = [_bbox_of(a) for a in group if _bbox_of(a)]
        merged_bbox = [
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        ] if boxes else None

        def _join(key):
            vals = [_field(a, key) for a in group]
            return " || ".join(vals) if len(group) > 1 else vals[0]

        rec = {
            "arrow_id": group[0].get("arrow_id", 0),
            "source_node": _join("source_node"),
            "condition": _join("condition") or "None",
            "target_node": _join("target_node"),
            bbox_key: merged_bbox,
        }
        if has_loc:
            rec["source_location"] = _join("source_location") or "None"
            rec["target_location"] = _join("target_location") or "None"
        merged.append(rec)

    return merged


# ── Whole-image resolution + drawing ────────────────────────────────────────────

def resolve_whole_image(folder: Path, data: dict, image_dir: Optional[Path]) -> Optional[Path]:
    img = data.get("image_path", "") or ""
    if img and Path(img).exists():
        return Path(img)
    cands = []
    if image_dir:
        cands += [image_dir / f"{folder.name}.png", image_dir / f"{folder.name}.jpeg"]
    cands += [folder / f"{folder.name}.png", folder / f"{folder.name}.jpeg"]
    for c in cands:
        if c.exists():
            return c
    return None


def draw_blue_box(img_path: Path, bbox: List[float], out_path: Path) -> bool:
    img = cv2.imread(str(img_path))
    if img is None or bbox is None:
        return False
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), BOX_LINE_WIDTH)  # BGR (255,0,0) = blue
    cv2.imwrite(str(out_path), img)
    return True


# ── Per-sample processing ───────────────────────────────────────────────────────

def process_sample(sample_dir: Path, image_dir: Optional[Path], iou_threshold: float) -> bool:
    json_files = [f for f in sample_dir.glob("*.json") if not f.name.endswith("_merged.json")]
    if not json_files:
        return False
    json_path = json_files[0]
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    arrows = data.get("arrows", [])
    if not arrows:
        return False

    whole_img = resolve_whole_image(sample_dir, data, image_dir)
    if whole_img is None:
        print(f"  skip {sample_dir.name}: whole image not found (pass --image-dir?)")
        return False

    merged = merge_arrows(arrows, iou_threshold)

    out_img_dir = sample_dir / "annotated_images_merged"
    out_img_dir.mkdir(exist_ok=True)
    for arr in merged:
        aid = arr.get("arrow_id", 0)
        out_img = out_img_dir / f"{sample_dir.name}_arrow_{aid}.png"
        if draw_blue_box(whole_img, _bbox_of(arr), out_img):
            arr["annotated_image_path"] = str(out_img.absolute())

    out = dict(data)
    out["arrows"] = merged
    merged_json = sample_dir / f"{json_path.stem}_merged.json"
    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if len(merged) < len(arrows):
        print(f"  {sample_dir.name}: {len(arrows)} -> {len(merged)} arrows (merged)")
    return True


def main():
    parser = argparse.ArgumentParser(description="[OPTIONAL] Merge overlapping arrowheads in intermediate data")
    parser.add_argument("--data-dir", required=True,
                        help="Intermediate-data directory (one folder per sample)")
    parser.add_argument("--image-dir", default=None,
                        help="Optional whole-image base dir (needed to redraw the merged box)")
    parser.add_argument("--iou-threshold", type=float, default=0.6,
                        help="Merge arrows whose arrowhead bboxes overlap with IoU >= this")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None

    print("=" * 60)
    print("[OPTIONAL] Merge overlapping arrowheads")
    print(f"  data dir : {data_dir}")
    print(f"  IoU >=   : {args.iou_threshold}")
    print("=" * 60)

    processed = 0
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if process_sample(folder, image_dir, args.iou_threshold):
            processed += 1
        if processed and processed % 200 == 0:
            print(f"    processed {processed}")

    print(f"\nDone. Wrote *_merged.json for {processed} samples.")


if __name__ == "__main__":
    main()
