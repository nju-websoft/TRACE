#!/usr/bin/env python3
"""
gen_coco_for_sam3.py

Generate COCO-format train/val data for the SAM3 arrowhead detection model.
Each source flowchart maps to one `images` record, holding the annotations for
all arrowhead bboxes in that image.

Data sources:
  FCB       : InkML files (FC_Train.txt / FC_Validation.txt), rendered and saved as PNG
  FlowLearn : SVG files + existing JPEG images
  FlowVQA   : SVG + PNG inside the data_4_training/flowvqa/ and flowvqa_test/ subfolders

Output to: <REPO_ROOT>/detection/datasets/
  sam3_fcb/        train.json / val.json  (images/ subfolder holds the rendered images)
  sam3_flowlearn/  train.json / val.json
  sam3_flowvqa/    train.json / val.json

Usage:
    python detection/gen_data/gen_coco_for_sam3.py [--datasets fcb flowlearn flowvqa]
"""

import os
import sys
import json
import math
import re
import argparse
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# ============================================================
# Path configuration
# ============================================================
DATASET_ROOT = Path("<DATA_ROOT>")
OUTPUT_ROOT  = Path("<REPO_ROOT>/detection/datasets")

# FCB
FCB_INKML_DIR = DATASET_ROOT / "FC_B_raw"
FCB_TRAIN_TXT = FCB_INKML_DIR / "FC_Train.txt"
FCB_VAL_TXT   = FCB_INKML_DIR / "FC_Validation.txt"

# FlowLearn — train/val are decided by which key-list json you point at (no auto-split)
FLOWLEARN_JPEG_DIR   = DATASET_ROOT / "flowlearn/mermaid_word/jpeg"
FLOWLEARN_SVG_DIR    = DATASET_ROOT / "flowlearn/mermaid_word/svg"
FLOWLEARN_TRAIN_JSON = DATASET_ROOT / "flowlearn/train.json"
FLOWLEARN_VAL_JSON   = DATASET_ROOT / "flowlearn/val.json"

# FlowVQA — train/val are decided by which folder the samples live in (no auto-split)
FLOWVQA_TRAIN_DIR = DATASET_ROOT / "data_4_training/flowvqa_train"
FLOWVQA_VAL_DIR   = DATASET_ROOT / "data_4_training/flowvqa_val"

# COCO category
ARROWHEAD_CAT = [{"id": 1, "name": "arrowhead", "supercategory": "arrow"}]

# FCB rendering parameters (kept consistent with gen_data_from_fcb.py)
INKML_MAX_DIM = 1333
INKML_PADDING = 20
INKML_LINE_W  = 2


# ============================================================
# Helper functions
# ============================================================

def xyxy_to_xywh(x1, y1, x2, y2):
    """[x1,y1,x2,y2] → COCO [x,y,w,h]"""
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    return [float(x1), float(y1), w, h]


def make_coco(images, annotations):
    return {
        "categories": ARROWHEAD_CAT,
        "images": images,
        "annotations": annotations,
    }


def save_coco(coco_dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(coco_dict, f, indent=2, ensure_ascii=False)
    n_img = len(coco_dict['images'])
    n_ann = len(coco_dict['annotations'])
    print(f"  Saved → {path}  ({n_img} images, {n_ann} annotations)")



# ============================================================
# FCB: InkML -> rendered image + arrowhead bboxes
# ============================================================

class InkMLParser:
    """Lightweight InkML parser: extracts arrowhead bboxes and renders the base image."""

    def __init__(self, inkml_path):
        tree = ET.parse(inkml_path)
        root = tree.getroot()
        # Strip namespaces
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]

        self.traces = {}
        self.entities = {}
        self.relations = []

        self._raw_min_x = float('inf')
        self._raw_max_x = float('-inf')
        self._raw_min_y = float('inf')
        self._raw_max_y = float('-inf')

        self._parse_traces(root)
        self._parse_entities(root)
        self._parse_relations(root)

        raw_w = max(self._raw_max_x - self._raw_min_x, 1)
        raw_h = max(self._raw_max_y - self._raw_min_y, 1)
        available = INKML_MAX_DIM - 2 * INKML_PADDING
        self.scale = available / max(raw_w, raw_h)
        self.img_w = int(raw_w * self.scale + 2 * INKML_PADDING)
        self.img_h = int(raw_h * self.scale + 2 * INKML_PADDING)

    def _parse_traces(self, root):
        for trace in root.findall('trace'):
            tid = trace.get('id')
            pts = []
            for p in trace.text.strip().split(','):
                c = p.strip().split()
                if len(c) >= 2:
                    x, y = float(c[0]), float(c[1])
                    pts.append([x, y])
                    self._raw_min_x = min(self._raw_min_x, x)
                    self._raw_max_x = max(self._raw_max_x, x)
                    self._raw_min_y = min(self._raw_min_y, y)
                    self._raw_max_y = max(self._raw_max_y, y)
            self.traces[tid] = np.array(pts, dtype=np.float32)

    def _parse_entities(self, root):
        symbols = root.find('symbols')
        if symbols is None:
            return
        for group in symbols.findall('traceGroup'):
            gid = group.get('id')
            truth = group.find("./annotation[@type='truth']")
            etype = truth.text if truth is not None else 'unknown'
            trace_ids = [tv.get('traceDataRef') for tv in group.findall('traceView')]
            head_ids = []
            head_elem = group.find('head')
            if head_elem is not None:
                head_ids = [tv.get('traceDataRef') for tv in head_elem.findall('traceView')]
            self.entities[gid] = {
                'type': etype,
                'trace_ids': trace_ids,
                'head_trace_ids': head_ids,
            }

    def _parse_relations(self, root):
        rels = root.find('relations')
        if rels is None:
            return
        for group in rels.findall('symbolGroup'):
            truth = group.find("./annotation[@type='truth']")
            if truth is None or truth.text != 'arrow_connection':
                continue
            refs = [v.get('symbolDataRef') for v in group.findall('symbolView')]
            arrow_id = None
            for rid in refs:
                if rid in self.entities and self.entities[rid]['type'] == 'arrow':
                    arrow_id = rid
                    break
            if arrow_id:
                self.relations.append({'arrow_id': arrow_id})

    def _raw_bbox(self, trace_ids):
        pts = [self.traces[tid] for tid in trace_ids if tid in self.traces]
        if not pts:
            return None
        pts = np.vstack(pts)
        return [float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max())]

    def _to_pixel(self, rx, ry):
        px = (rx - self._raw_min_x) * self.scale + INKML_PADDING
        py = (ry - self._raw_min_y) * self.scale + INKML_PADDING
        return int(px), int(py)

    def render_base(self) -> np.ndarray:
        """Render the raw flowchart image (white background, black strokes, no annotations)."""
        img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        for pts in self.traces.values():
            if len(pts) < 2:
                continue
            scaled = np.array(
                [self._to_pixel(x, y) for x, y in pts], dtype=np.int32
            ).reshape(-1, 1, 2)
            cv2.polylines(img, [scaled], False, (0, 0, 0), INKML_LINE_W, lineType=cv2.LINE_AA)
        return img

    def get_arrowhead_bboxes(self):
        """Return the pixel bboxes of all arrowheads, as [x1, y1, x2, y2]."""
        bboxes = []
        for rel in self.relations:
            arrow = self.entities[rel['arrow_id']]
            head_ids = arrow['head_trace_ids'] or arrow['trace_ids']
            raw = self._raw_bbox(head_ids)
            if raw is None:
                continue
            bx1, by1 = self._to_pixel(raw[0], raw[1])
            bx2, by2 = self._to_pixel(raw[2], raw[3])
            m = 5
            bx1 = max(0, bx1 - m)
            by1 = max(0, by1 - m)
            bx2 = min(self.img_w, bx2 + m)
            by2 = min(self.img_h, by2 + m)
            if bx2 > bx1 and by2 > by1:
                bboxes.append([bx1, by1, bx2, by2])
        return bboxes


def gen_fcb_coco(split_txt: Path, images_out_dir: Path, split_name: str) -> dict:
    """
    Build the FCB COCO dict from FC_Train.txt / FC_Validation.txt.
    The rendered base images are saved to images_out_dir.
    """
    images_out_dir.mkdir(parents=True, exist_ok=True)

    with open(split_txt) as f:
        keys = [l.strip().replace('.inkml', '') for l in f if l.strip()]

    coco_imgs, coco_anns = [], []
    img_id = ann_id = 0
    skipped = 0

    for key in keys:
        inkml_path = FCB_INKML_DIR / f"{key}.inkml"
        if not inkml_path.exists():
            skipped += 1
            continue
        try:
            parser = InkMLParser(str(inkml_path))
            bboxes = parser.get_arrowhead_bboxes()
            if not bboxes:
                skipped += 1
                continue

            img_file = images_out_dir / f"{key}.png"
            if not img_file.exists():
                cv2.imwrite(str(img_file), parser.render_base())

            img_id += 1
            coco_imgs.append({
                "id": img_id,
                "file_name": str(img_file.absolute()),
                "width": parser.img_w,
                "height": parser.img_h,
            })
            for bbox in bboxes:
                ann_id += 1
                xywh = xyxy_to_xywh(*bbox)
                coco_anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": xywh,
                    "area": xywh[2] * xywh[3],
                    "segmentation": [],
                    "iscrowd": 0,
                })
        except Exception as e:
            print(f"  [FCB] error {key}: {e}")
            skipped += 1

    print(f"  [FCB {split_name}] {img_id} images, {ann_id} annotations, {skipped} skipped")
    return make_coco(coco_imgs, coco_anns)


# ============================================================
# FlowLearn: SVG -> arrowhead bboxes + existing JPEG
# ============================================================

def _flowlearn_arrowhead_bbox(d_string, off_x, off_y):
    """Compute the arrowhead bbox from the SVG path `d` attribute (matches gen_data_from_flowlearn.py)."""
    points = [float(p) for p in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', d_string)]
    if len(points) < 4:
        return None
    tip_x, tip_y = points[-2], points[-1]
    prev_x, prev_y = points[-4], points[-3]
    dx, dy = tip_x - prev_x, tip_y - prev_y
    angle = math.atan2(dy, dx)
    arrow_len, half_w = 10.0, 5.0
    local = [(0, 0), (-arrow_len, -half_w), (-arrow_len, half_w)]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    xs, ys = [], []
    for lx, ly in local:
        rx = lx * cos_a - ly * sin_a
        ry = lx * sin_a + ly * cos_a
        xs.append((rx + tip_x) - off_x)
        ys.append((ry + tip_y) - off_y)
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def parse_flowlearn_bboxes(svg_content: str):
    """Parse a FlowLearn SVG and return all arrowhead bboxes."""
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ns = {'svg': 'http://www.w3.org/2000/svg', 'xhtml': 'http://www.w3.org/1999/xhtml'}
    root = ET.fromstring(svg_content)

    viewbox = root.get('viewBox', '')
    off_x = off_y = 0.0
    if viewbox:
        parts = [float(x) for x in re.split(r'[,\s]+', viewbox.strip())]
        if len(parts) >= 2:
            off_x, off_y = parts[0], parts[1]

    bboxes = []
    for path in root.findall('.//svg:path', ns):
        if 'flowchart-link' not in path.get('class', ''):
            continue
        bbox = _flowlearn_arrowhead_bbox(path.get('d', ''), off_x, off_y)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


def gen_flowlearn_coco(keys: list, split_name: str) -> dict:
    """Build the FlowLearn COCO dict from a list of file names."""

    coco_imgs, coco_anns = [], []
    img_id = ann_id = skipped = 0

    for key in keys:
        base = Path(key).stem
        jpeg_path = FLOWLEARN_JPEG_DIR / f"{base}.jpeg"
        if not jpeg_path.exists():
            jpeg_path = FLOWLEARN_JPEG_DIR / f"{base}.jpg"
        svg_path = FLOWLEARN_SVG_DIR / f"{base}.svg"

        if not jpeg_path.exists() or not svg_path.exists():
            skipped += 1
            continue

        try:
            with open(svg_path, encoding='utf-8') as f:
                svg = f.read()
            bboxes = parse_flowlearn_bboxes(svg)
            if not bboxes:
                skipped += 1
                continue

            with Image.open(jpeg_path) as img:
                w, h = img.size

            img_id += 1
            coco_imgs.append({
                "id": img_id,
                "file_name": str(jpeg_path.absolute()),
                "width": w,
                "height": h,
            })
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(w), x2), min(float(h), y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                ann_id += 1
                xywh = xyxy_to_xywh(x1, y1, x2, y2)
                coco_anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": xywh,
                    "area": xywh[2] * xywh[3],
                    "segmentation": [],
                    "iscrowd": 0,
                })
        except Exception as e:
            print(f"  [FlowLearn] error {base}: {e}")
            skipped += 1

    print(f"  [FlowLearn {split_name}] {img_id} images, {ann_id} annotations, {skipped} skipped")
    return make_coco(coco_imgs, coco_anns)


# ============================================================
# FlowVQA: SVG + PNG subfolders
# ============================================================

def _parse_svg_last_segment(d_str):
    nums = [float(n) for n in re.findall(r'-?[\d\.]+', d_str)]
    if len(nums) >= 4:
        return (nums[-4], nums[-3]), (nums[-2], nums[-1])
    if len(nums) >= 2:
        return (0.0, 0.0), (nums[-2], nums[-1])
    return (0.0, 0.0), (0.0, 0.0)


def _flowvqa_arrowhead_bbox(prev_pt, end_pt, arrow_len=10, arrow_width=10, padding=5):
    dx = end_pt[0] - prev_pt[0]
    dy = end_pt[1] - prev_pt[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return [end_pt[0] - padding, end_pt[1] - padding,
                end_pt[0] + padding, end_pt[1] + padding]
    ux, uy = dx / length, dy / length
    vx, vy = -uy, ux
    half_w = arrow_width / 2
    p1 = end_pt
    base_cx = end_pt[0] - ux * arrow_len
    base_cy = end_pt[1] - uy * arrow_len
    p2 = (base_cx + vx * half_w, base_cy + vy * half_w)
    p3 = (base_cx - vx * half_w, base_cy - vy * half_w)
    xs = [p1[0], p2[0], p3[0]]
    ys = [p1[1], p2[1], p3[1]]
    return [min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding]


def parse_flowvqa_bboxes(svg_content: str):
    """Parse a FlowVQA SVG and return all arrowhead bboxes."""
    svg_content = re.sub(r' xmlns="[^"]+"', '', svg_content, count=1)
    svg_content = re.sub(r' xmlns:xlink="[^"]+"', '', svg_content, count=1)
    root = ET.fromstring(svg_content)
    bboxes = []
    for path in root.findall('.//path'):
        if 'flowchart-link' not in path.get('class', ''):
            continue
        prev_pt, end_pt = _parse_svg_last_segment(path.get('d', ''))
        bboxes.append(_flowvqa_arrowhead_bbox(prev_pt, end_pt))
    return bboxes


def gen_flowvqa_coco(subfolders: list, split_name: str) -> dict:
    """
    Build the FlowVQA COCO dict from a list of subfolders (each holds one SVG and one PNG).
    """
    coco_imgs, coco_anns = [], []
    img_id = ann_id = skipped = 0

    for folder in subfolders:
        svg_files = list(folder.glob('*.svg'))
        png_files = list(folder.glob('*.png'))

        if not svg_files or not png_files:
            skipped += 1
            continue

        try:
            with open(svg_files[0], encoding='utf-8') as f:
                svg = f.read()
            bboxes = parse_flowvqa_bboxes(svg)
            if not bboxes:
                skipped += 1
                continue

            with Image.open(png_files[0]) as img:
                w, h = img.size

            img_id += 1
            coco_imgs.append({
                "id": img_id,
                "file_name": str(png_files[0].absolute()),
                "width": w,
                "height": h,
            })
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(w), x2), min(float(h), y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                ann_id += 1
                xywh = xyxy_to_xywh(x1, y1, x2, y2)
                coco_anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": xywh,
                    "area": xywh[2] * xywh[3],
                    "segmentation": [],
                    "iscrowd": 0,
                })
        except Exception as e:
            print(f"  [FlowVQA] error {folder.name}: {e}")
            skipped += 1

    print(f"  [FlowVQA {split_name}] {img_id} images, {ann_id} annotations, {skipped} skipped")
    return make_coco(coco_imgs, coco_anns)


# ============================================================
# BPMN: extract arrowhead bboxes from the training-data JSON
# ============================================================

BPMN_DATA_DIR = DATASET_ROOT / "data_4_training" / "bpmn"
BPMN_PNG_DIRS = {
    "train": DATASET_ROOT / "bpmn" / "new_train_images",
    "dev":   DATASET_ROOT / "bpmn" / "new_dev_images",
    "test":  DATASET_ROOT / "bpmn" / "new_test_images",
}


def _bpmn_find_png(image_name: str) -> Path | None:
    """Find the matching PNG for an image_name (e.g. 'process_001')."""
    for png_dir in BPMN_PNG_DIRS.values():
        p = png_dir / f"{image_name}.png"
        if p.exists():
            return p
    return None


def gen_bpmn_coco(sample_dirs: list, split_name: str) -> dict:
    coco_imgs, coco_anns = [], []
    img_id = ann_id = skipped = 0

    for folder in sample_dirs:
        json_path = folder / f"{folder.name}.json"
        if not json_path.exists():
            skipped += 1
            continue
        try:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
            png_path = _bpmn_find_png(data["image_name"])
            if png_path is None:
                skipped += 1
                continue
            with Image.open(png_path) as img:
                w, h = img.size

            bboxes = [a["matched_arrowhead_bbox"] for a in data["arrows"]
                      if "matched_arrowhead_bbox" in a]
            if not bboxes:
                skipped += 1
                continue

            img_id += 1
            coco_imgs.append({
                "id": img_id,
                "file_name": str(png_path.absolute()),
                "width": w, "height": h,
            })
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(w), x2), min(float(h), y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                ann_id += 1
                xywh = xyxy_to_xywh(x1, y1, x2, y2)
                coco_anns.append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": xywh, "area": xywh[2] * xywh[3],
                    "segmentation": [], "iscrowd": 0,
                })
        except Exception as e:
            print(f"  [BPMN] error {folder.name}: {e}")
            skipped += 1

    print(f"  [BPMN {split_name}] {img_id} images, {ann_id} annotations, {skipped} skipped")
    return make_coco(coco_imgs, coco_anns)


# ============================================================
# FlowGen: extract arrowhead bboxes from the training-data JSON
# ============================================================

FLOWGEN_ROOT = Path("<DATA_ROOT>/flowgen")
FLOWGEN_DATA_ROOT = DATASET_ROOT / "data_4_training"


def _flowgen_find_png(sample_id: str, difficulty: str) -> Path | None:
    """Find the downscaled PNG for a sample_id.
    sample_id format: {type}_{diff}_{stem}  e.g. diagrams_medium_diagrams_101
    Downscaled PNG: flowgen/train_{type}_{diff}_png/{stem}.png
    """
    parts = sample_id.split('_', 2)  # ['diagrams', 'medium', 'diagrams_101']
    if len(parts) < 3:
        return None
    dtype, _, stem = parts
    p = FLOWGEN_ROOT / f"train_{dtype}_{difficulty}_png" / f"{stem}.png"
    return p if p.exists() else None


def gen_flowgen_coco(sample_dirs: list, split_name: str, difficulty: str) -> dict:
    coco_imgs, coco_anns = [], []
    img_id = ann_id = skipped = 0

    for folder in sample_dirs:
        json_path = folder / f"{folder.name}.json"
        if not json_path.exists():
            skipped += 1
            continue
        try:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
            png_path = _flowgen_find_png(data["sample_id"], difficulty)
            if png_path is None:
                skipped += 1
                continue
            with Image.open(png_path) as img:
                w, h = img.size

            bboxes = [a["arrowhead_bbox"] for a in data["arrows"]
                      if "arrowhead_bbox" in a]
            if not bboxes:
                skipped += 1
                continue

            img_id += 1
            coco_imgs.append({
                "id": img_id,
                "file_name": str(png_path.absolute()),
                "width": w, "height": h,
            })
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(w), x2), min(float(h), y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                ann_id += 1
                xywh = xyxy_to_xywh(x1, y1, x2, y2)
                coco_anns.append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": xywh, "area": xywh[2] * xywh[3],
                    "segmentation": [], "iscrowd": 0,
                })
        except Exception as e:
            print(f"  [FlowGen] error {folder.name}: {e}")
            skipped += 1

    print(f"  [FlowGen {split_name}] {img_id} images, {ann_id} annotations, {skipped} skipped")
    return make_coco(coco_imgs, coco_anns)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate COCO-format training data for SAM3 arrowhead detection')
    parser.add_argument('--datasets', nargs='+',
                        default=['fcb', 'flowlearn', 'flowvqa'],
                        choices=['fcb', 'flowlearn', 'flowvqa',
                                 'bpmn', 'flowgen_easy', 'flowgen_medium', 'flowgen_hard'],
                        help='Datasets to process')
    args = parser.parse_args()

    if 'fcb' in args.datasets:
        print("\n[FCB]")
        out = OUTPUT_ROOT / "sam3_fcb"
        # FCB has an official val split, use it directly
        train = gen_fcb_coco(FCB_TRAIN_TXT, out / "images" / "train", "train")
        val   = gen_fcb_coco(FCB_VAL_TXT,   out / "images" / "val",   "val")
        save_coco(train, out / "train.json")
        save_coco(val,   out / "val.json")

    if 'flowlearn' in args.datasets:
        print("\n[FlowLearn]")
        out = OUTPUT_ROOT / "sam3_flowlearn"

        def _load_keys(p):
            with open(p) as f:
                d = json.load(f)
            return list(d.keys()) if isinstance(d, dict) else list(d)

        # No auto-split: train/val keys come from the two json files the user provides.
        train_keys = _load_keys(FLOWLEARN_TRAIN_JSON)
        val_keys   = _load_keys(FLOWLEARN_VAL_JSON) if FLOWLEARN_VAL_JSON.exists() else []
        print(f"  train: {len(train_keys)} keys, val: {len(val_keys)} keys")
        train = gen_flowlearn_coco(train_keys, "train")
        val   = gen_flowlearn_coco(val_keys,   "val") if val_keys else make_coco([], [])
        save_coco(train, out / "train.json")
        save_coco(val,   out / "val.json")

    if 'flowvqa' in args.datasets:
        print("\n[FlowVQA]")
        out = OUTPUT_ROOT / "sam3_flowvqa"
        if not FLOWVQA_TRAIN_DIR.exists():
            print(f"  [FlowVQA] train directory not found: {FLOWVQA_TRAIN_DIR}")
        else:
            # No auto-split: train/val come from separate folders the user provides.
            train_folders = sorted([d for d in FLOWVQA_TRAIN_DIR.iterdir() if d.is_dir()])
            val_folders   = sorted([d for d in FLOWVQA_VAL_DIR.iterdir() if d.is_dir()]) if FLOWVQA_VAL_DIR.exists() else []
            print(f"  train: {len(train_folders)}, val: {len(val_folders)}")
            train = gen_flowvqa_coco(train_folders, "train")
            val   = gen_flowvqa_coco(val_folders,   "val") if val_folders else make_coco([], [])
            save_coco(train, out / "train.json")
            save_coco(val,   out / "val.json")

    if 'bpmn' in args.datasets:
        print("\n[BPMN]")
        out = OUTPUT_ROOT / "sam3_bpmn"
        bpmn_val_dir = FLOWGEN_DATA_ROOT / "bpmn_test"
        train_folders = sorted([d for d in BPMN_DATA_DIR.iterdir() if d.is_dir()])
        val_folders   = sorted([d for d in bpmn_val_dir.iterdir() if d.is_dir()])
        print(f"  train: {len(train_folders)}, val: {len(val_folders)}")
        train = gen_bpmn_coco(train_folders, "train")
        val   = gen_bpmn_coco(val_folders,   "val")
        save_coco(train, out / "train.json")
        save_coco(val,   out / "val.json")

    for diff in ['easy', 'medium', 'hard']:
        ds_name = f'flowgen_{diff}'
        if ds_name not in args.datasets:
            continue
        print(f"\n[FlowGen {diff}]")
        out = OUTPUT_ROOT / f"sam3_flowgen_{diff}"
        # No auto-split: the user decides train/val by which folder the samples live in.
        train_dir = FLOWGEN_DATA_ROOT / f"flowgen_train_{diff}"
        val_dir   = FLOWGEN_DATA_ROOT / f"flowgen_val_{diff}"
        if not train_dir.exists():
            print(f"  train directory not found: {train_dir}")
            continue
        train_folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        val_folders   = sorted([d for d in val_dir.iterdir() if d.is_dir()]) if val_dir.exists() else []
        print(f"  train: {len(train_folders)}, val: {len(val_folders)}")
        train = gen_flowgen_coco(train_folders, "train", diff)
        val   = gen_flowgen_coco(val_folders,   "val",   diff) if val_folders else make_coco([], [])
        save_coco(train, out / "train.json")
        save_coco(val,   out / "val.json")

    print("\n✅ All datasets generated.")
    print(f"Output directory: {OUTPUT_ROOT}")


if __name__ == '__main__':
    main()
