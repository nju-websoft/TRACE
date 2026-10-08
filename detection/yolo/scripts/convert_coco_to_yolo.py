#!/usr/bin/env python3
"""
COCO -> YOLO data format conversion.

Input:  detection/datasets/sam3_<ds>/{train,val}.json (COCO format)
Output: detection/yolo/data/<ds>/
    images/{train,val}/<symlinks to source images>
    labels/{train,val}/<stem>.txt   (YOLO format: cls cx cy w h, all normalized)
    data.yaml                       (Ultralytics dataset yaml)

Idempotent: existing outputs are overwritten.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x, y, w, h] (top-left) -> normalized YOLO [cx, cy, w, h]."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def convert_split(coco_json, dst_img_dir, dst_lbl_dir, force=False):
    with open(coco_json, "r") as f:
        data = json.load(f)

    # file_name in the COCO json is relative to the json's own directory
    coco_root = Path(coco_json).resolve().parent

    cats = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_yolo_idx = {c["id"]: i for i, c in enumerate(cats)}

    # Collect annotations by image_id
    ann_by_img = {}
    for a in data["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_img, n_ann, n_skip = 0, 0, 0
    for img in data["images"]:
        src_path = Path(img["file_name"])
        if not src_path.is_absolute():
            # Resolve relative to the COCO json's directory
            src_path = coco_root / src_path
        if not src_path.exists():
            print(f"  [skip] missing image: {src_path}")
            n_skip += 1
            continue

        img_w, img_h = img["width"], img["height"]
        stem = src_path.stem

        # Create an image symlink (same name -> avoid overwrite)
        link = dst_img_dir / src_path.name
        if link.exists() or link.is_symlink():
            if force:
                link.unlink()
            else:
                # Already exists; assume it points to the right file
                pass
        if not link.exists():
            try:
                link.symlink_to(src_path)
            except OSError as e:
                print(f"  [warn] symlink failed {src_path} -> {link}: {e}")
                continue

        # Write labels
        label_path = dst_lbl_dir / f"{stem}.txt"
        anns = ann_by_img.get(img["id"], [])
        lines = []
        for a in anns:
            cls = cat_id_to_yolo_idx.get(a["category_id"])
            if cls is None:
                continue
            cx, cy, nw, nh = coco_bbox_to_yolo(a["bbox"], img_w, img_h)
            # clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_ann += 1
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        n_img += 1

    return n_img, n_ann, n_skip, [c["name"] for c in cats]


def write_data_yaml(yaml_path, root, names):
    rel_train_img = "images/train"
    rel_val_img = "images/val"
    content = (
        f"path: {root}\n"
        f"train: {rel_train_img}\n"
        f"val: {rel_val_img}\n"
        f"nc: {len(names)}\n"
        f"names:\n"
    )
    for i, n in enumerate(names):
        content += f"  {i}: {n}\n"
    with open(yaml_path, "w") as f:
        f.write(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="<REPO_ROOT>/detection/datasets",
                    help="Root of the source COCO data (contains sam3_<ds>/{train,val}.json)")
    ap.add_argument("--dst-root", default="<REPO_ROOT>/detection/yolo/data",
                    help="Root of the YOLO output")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="Dataset names to convert (without the sam3_ prefix)")
    ap.add_argument("--force", action="store_true", help="Rebuild symlinks/labels")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for ds in args.datasets:
        src_dir = src_root / f"sam3_{ds}"
        train_json = src_dir / "train.json"
        val_json = src_dir / "val.json"
        if not train_json.exists() or not val_json.exists():
            print(f"[FATAL] missing train/val JSON: {src_dir}")
            sys.exit(1)

        ds_root = dst_root / ds
        if args.force and ds_root.exists():
            shutil.rmtree(ds_root)

        print(f"\n=== {ds} ===")
        n_train, a_train, s_train, names = convert_split(
            train_json, ds_root / "images" / "train",
            ds_root / "labels" / "train", force=args.force
        )
        print(f"  train: imgs={n_train}  ann={a_train}  skipped={s_train}")
        n_val, a_val, s_val, _ = convert_split(
            val_json, ds_root / "images" / "val",
            ds_root / "labels" / "val", force=args.force
        )
        print(f"  val:   imgs={n_val}    ann={a_val}    skipped={s_val}")

        yaml_path = ds_root / "data.yaml"
        write_data_yaml(yaml_path, str(ds_root), names)
        print(f"  data.yaml: {yaml_path} (names={names})")

        summary.append((ds, n_train, n_val, a_train, a_val))

    print("\nSummary:")
    print(f"  {'dataset':16s} {'train_img':>10s} {'val_img':>10s} {'train_ann':>10s} {'val_ann':>10s}")
    for ds, ti, vi, ta, va in summary:
        print(f"  {ds:16s} {ti:>10d} {vi:>10d} {ta:>10d} {va:>10d}")


if __name__ == "__main__":
    main()
