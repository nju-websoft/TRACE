"""
Generate BPMN training data (via the SVG route).

Flow:
  1. Convert BPMN XML -> SVG with `npx bpmn-to-image`
  2. Convert SVG -> PNG with cairosvg (1:1 coordinate mapping)
  3. Parse the SVG link paths to get exact arrowhead pixel coordinates
  4. Parse the BPMN XML for semantics (source/target node names, condition, pool/lane)
  5. Draw a blue box on the arrowhead in the PNG and emit training records

Usage:
    python gen_data_from_bpmn.py [--splits train dev test] [--output-json bpmn_arrow_training_all.json]
"""

import argparse
import io
import json
import math
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw

# ── Paths ─────────────────────────────────────────────────────────────────────
BPMN_ROOT   = Path("<DATA_ROOT>/bpmn")
XML_ROOT    = BPMN_ROOT / "BPMN_XML_Files"
OUTPUT_ROOT = Path("<DATA_ROOT>/data_4_training/bpmn_test")

# ── BPMN namespaces ───────────────────────────────────────────────────────────
BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"

FLOW_NODE_TYPES = {
    "startEvent", "endEvent",
    "task", "userTask", "serviceTask", "sendTask", "receiveTask",
    "manualTask", "businessRuleTask", "scriptTask",
    "subProcess", "callActivity",
    "exclusiveGateway", "parallelGateway", "inclusiveGateway",
    "eventBasedGateway", "complexGateway",
    "intermediateCatchEvent", "intermediateThrowEvent", "boundaryEvent",
}

# ── Blue box parameters ───────────────────────────────────────────────────────
BOX_WIDTH   = 3    # rectangle line width (pixels)
BOX_PADDING = 5    # padding around arrowhead triangle

# ── XML helpers ───────────────────────────────────────────────────────────────

def ns_tag(namespace: str, local: str) -> str:
    return f"{{{namespace}}}{local}"


def local_tag(element) -> str:
    t = element.tag
    return t.split("}")[-1] if "}" in t else t


def clean(text) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


# ── BPMN XML parser (semantic info only) ─────────────────────────────────────

def parse_bpmn_xml(xml_path: Path) -> dict:
    """
    Parse BPMN XML for semantic info: node names/types, pool/lane membership,
    and connection flow_id → (source, target, condition).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. Pool mapping
    pool_of_process: dict[str, str] = {}
    for collab in root.iter(ns_tag(BPMN_NS, "collaboration")):
        for part in collab.findall(ns_tag(BPMN_NS, "participant")):
            pname = clean(part.get("name", ""))
            proc = part.get("processRef", "")
            if proc and pname:
                pool_of_process[proc] = pname

    # 2. Lane → node mapping
    node_pool: dict[str, str | None] = {}
    node_lane: dict[str, str] = {}

    for process in root.iter(ns_tag(BPMN_NS, "process")):
        proc_id = process.get("id", "")
        pool_name = pool_of_process.get(proc_id) or clean(process.get("name", "")) or None
        for lane in process.iter(ns_tag(BPMN_NS, "lane")):
            lane_name = clean(lane.get("name", ""))
            for ref in lane.findall(ns_tag(BPMN_NS, "flowNodeRef")):
                nid = (ref.text or "").strip()
                if nid:
                    node_pool[nid] = pool_name
                    if lane_name:
                        node_lane[nid] = lane_name

    # 3. Node registry
    nodes: dict[str, dict] = {}
    for process in root.iter(ns_tag(BPMN_NS, "process")):
        proc_id = process.get("id", "")
        pool_name = pool_of_process.get(proc_id) or clean(process.get("name", "")) or None
        for elem in process:
            lname = local_tag(elem)
            if lname not in FLOW_NODE_TYPES:
                continue
            nid = elem.get("id", "")
            nname = clean(elem.get("name", "")) or lname
            nodes[nid] = {
                "name": nname,
                "type": lname,
                "pool": node_pool.get(nid, pool_name),
                "lane": node_lane.get(nid, ""),
            }
            if nid not in node_pool:
                node_pool[nid] = pool_name

    # 4. Connections keyed by flow_id
    def node_info(nid: str) -> dict:
        return nodes.get(nid, {
            "name": nid, "type": "unknown",
            "pool": node_pool.get(nid), "lane": node_lane.get(nid, ""),
        })

    connections: dict[str, dict] = {}

    for flow_tag_name in ("sequenceFlow", "messageFlow"):
        for sf in root.iter(ns_tag(BPMN_NS, flow_tag_name)):
            fid = sf.get("id", "")
            src = sf.get("sourceRef", "")
            tgt = sf.get("targetRef", "")
            cond = clean(sf.get("name", ""))
            s, t = node_info(src), node_info(tgt)
            connections[fid] = {
                "flow_id":     fid,
                "source":      s["name"],
                "source_pool": s["pool"],
                "source_lane": s["lane"],
                "condition":   cond,
                "target":      t["name"],
                "target_pool": t["pool"],
                "target_lane": t["lane"],
            }

    return {"nodes": nodes, "connections": connections}


# ── SVG parser (arrowhead coordinates) ───────────────────────────────────────

def parse_svg_connections(svg_path: Path) -> tuple[dict, tuple]:
    """
    Parse bpmn-to-image SVG to extract arrowhead coordinates.

    Returns:
      arrows: {flow_id: (tip_x, tip_y, prev_x, prev_y)}  in pixel coords
      viewbox: (vb_x, vb_y, vb_w, vb_h)
    """
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    # Parse viewBox
    vb_match = re.search(r'viewBox="([^"]+)"', svg_content)
    vb_x, vb_y, vb_w, vb_h = 0.0, 0.0, 100.0, 100.0
    if vb_match:
        parts = [float(x) for x in re.split(r'[,\s]+', vb_match.group(1).strip())]
        if len(parts) >= 4:
            vb_x, vb_y, vb_w, vb_h = parts[:4]

    # Split by connection groups
    segments = svg_content.split("djs-element djs-connection")
    arrows: dict[str, tuple] = {}

    for seg in segments[1:]:
        # Get element ID (= BPMN flow ID)
        eid_match = re.search(r'data-element-id="([^"]+)"', seg)
        if not eid_match:
            continue
        flow_id = eid_match.group(1)

        # Get the actual path from djs-hit-stroke (contains real route coords)
        # Format: <path d="M90,450L180,450" class="djs-hit djs-hit-stroke" .../>
        path_match = re.search(
            r'<path\s+d="([^"]+)"[^>]*class="djs-hit djs-hit-stroke"', seg
        )
        if not path_match:
            # Fallback: visual path with data-corner-radius
            path_match = re.search(
                r'data-corner-radius="[^"]*"[^>]*d="([^"]+)"', seg
            )
        if not path_match:
            continue

        d_str = path_match.group(1)
        nums = re.findall(r'[-+]?\d*\.?\d+', d_str)
        nums = [float(n) for n in nums]

        if len(nums) < 4:
            continue

        # Last point = tip, second-to-last = prev
        tip_svg_x, tip_svg_y = nums[-2], nums[-1]
        prev_svg_x, prev_svg_y = nums[-4], nums[-3]

        # Convert SVG coords to pixel coords (1:1 when using cairosvg)
        tip_px  = tip_svg_x - vb_x
        tip_py  = tip_svg_y - vb_y
        prev_px = prev_svg_x - vb_x
        prev_py = prev_svg_y - vb_y

        arrows[flow_id] = (tip_px, tip_py, prev_px, prev_py)

    return arrows, (vb_x, vb_y, vb_w, vb_h)


def svg_to_png(svg_path: Path) -> Image.Image:
    """Convert SVG to PNG using cairosvg, preserving exact viewBox dimensions."""
    with open(svg_path, "rb") as f:
        svg_data = f.read()
    png_data = cairosvg.svg2png(bytestring=svg_data)
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")
    # Composite onto white background (SVG may have transparent regions)
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img)
    return bg.convert("RGB")


# ── Image annotation ─────────────────────────────────────────────────────────

def calculate_arrowhead_bbox(tip_x, tip_y, prev_x, prev_y):
    """
    Calculate bounding box of the arrowhead triangle.
    bpmn-js arrowheads: ~10px long, ~5px half-width.
    """
    dx = tip_x - prev_x
    dy = tip_y - prev_y
    length = math.hypot(dx, dy)
    if length == 0:
        return (tip_x - 8, tip_y - 8, tip_x + 8, tip_y + 8)

    angle = math.atan2(dy, dx)
    arrow_len = 10.0
    arrow_half_w = 5.0

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Triangle vertices: tip, and two base corners
    local_pts = [(0, 0), (-arrow_len, -arrow_half_w), (-arrow_len, arrow_half_w)]
    xs, ys = [], []
    for lx, ly in local_pts:
        rx = lx * cos_a - ly * sin_a + tip_x
        ry = lx * sin_a + ly * cos_a + tip_y
        xs.append(rx)
        ys.append(ry)

    return (min(xs), min(ys), max(xs), max(ys))


def annotate_arrowhead(img: Image.Image, tip_x, tip_y, prev_x, prev_y) -> Image.Image:
    """Draw a blue bounding box around the arrowhead."""
    x1, y1, x2, y2 = calculate_arrowhead_bbox(tip_x, tip_y, prev_x, prev_y)
    img_w, img_h = img.size

    # Add padding and clamp
    x1 = max(0, round(x1 - BOX_PADDING))
    y1 = max(0, round(y1 - BOX_PADDING))
    x2 = min(img_w - 1, round(x2 + BOX_PADDING))
    y2 = min(img_h - 1, round(y2 + BOX_PADDING))

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=BOX_WIDTH)
    return annotated


# ── Immediate container helper ────────────────────────────────────────────────

def immediate_container(conn_info: dict, source_or_target: str) -> str | None:
    lane = conn_info.get(f"{source_or_target}_lane", "")
    pool = conn_info.get(f"{source_or_target}_pool")
    if lane:
        return lane
    if pool:
        return pool
    return None


# ── Batch SVG conversion ─────────────────────────────────────────────────────

def batch_convert_to_svg(xml_dir: Path, svg_dir: Path) -> int:
    """
    Convert all .bpmn files in xml_dir to SVG using npx bpmn-to-image.
    Returns number of conversions attempted.
    """
    svg_dir.mkdir(parents=True, exist_ok=True)
    bpmn_files = sorted(xml_dir.glob("*.bpmn"))

    # Build conversion args: each file as input:output pair
    # bpmn-to-image accepts multiple pairs in one call
    pairs = []
    for bpmn_path in bpmn_files:
        svg_path = svg_dir / f"{bpmn_path.stem}.svg"
        if svg_path.exists():
            continue  # skip already converted
        pairs.append(f"{bpmn_path}:{svg_path}")

    if not pairs:
        print(f"  All {len(bpmn_files)} SVGs already exist, skipping conversion")
        return 0

    print(f"  Converting {len(pairs)} BPMN files to SVG...")

    # Process in batches to avoid command line length limits
    batch_size = 20
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        try:
            subprocess.run(
                ["npx", "bpmn-to-image"] + batch,
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            print(f"  ⚠ Timeout on batch {i // batch_size + 1}")
        except Exception as e:
            print(f"  ✗ Conversion error: {e}")

        if (i + batch_size) % 100 == 0:
            print(f"    converted {min(i + batch_size, len(pairs))}/{len(pairs)}")

    return len(pairs)


# ── Main generation ───────────────────────────────────────────────────────────

def process_split(split: str) -> tuple[int, int, int]:
    """Process all BPMN files in one split. Returns (processed, skipped, arrows)."""
    xml_dir = XML_ROOT / split
    svg_dir = BPMN_ROOT / f"{split}_svg"
    out_dir = OUTPUT_ROOT

    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Batch convert BPMN → SVG
    batch_convert_to_svg(xml_dir, svg_dir)

    processed = skipped = total_arrows = 0
    xml_files = sorted(xml_dir.glob("*.bpmn"))
    print(f"\n[{split}] {len(xml_files)} BPMN files found")

    for xml_path in xml_files:
        pid = xml_path.stem  # e.g. "process_103"
        svg_path = svg_dir / f"{pid}.svg"

        if not svg_path.exists():
            print(f"  ⚠ no SVG for {pid}, skipping")
            skipped += 1
            continue

        # Parse BPMN XML for semantic info
        try:
            parsed = parse_bpmn_xml(xml_path)
        except Exception as e:
            print(f"  ✗ XML parse error {pid}: {e}")
            skipped += 1
            continue

        # Parse SVG for arrowhead coordinates
        try:
            svg_arrows, viewbox = parse_svg_connections(svg_path)
        except Exception as e:
            print(f"  ✗ SVG parse error {pid}: {e}")
            skipped += 1
            continue

        if not svg_arrows:
            skipped += 1
            continue

        # Convert SVG → PNG
        try:
            png_img = svg_to_png(svg_path)
        except Exception as e:
            print(f"  ✗ SVG→PNG error {pid}: {e}")
            skipped += 1
            continue

        img_w, img_h = png_img.size

        # Per-process output dirs
        proc_out_dir = out_dir / pid
        ann_img_dir = proc_out_dir / "annotated_images"
        ann_img_dir.mkdir(parents=True, exist_ok=True)

        # Match SVG arrows to BPMN XML connections by flow_id
        xml_connections = parsed["connections"]
        arrow_idx = 0
        proc_arrows = []  # per-process arrow data for JSON

        for flow_id, (tip_px, tip_py, prev_px, prev_py) in svg_arrows.items():
            conn = xml_connections.get(flow_id)
            if conn is None:
                continue  # SVG has edge not in our flow list (e.g. association)

            arrow_idx += 1

            # Clamp to image bounds
            tip_px = max(0, min(img_w - 1, tip_px))
            tip_py = max(0, min(img_h - 1, tip_py))

            # Calculate bbox for JSON
            bbox = calculate_arrowhead_bbox(tip_px, tip_py, prev_px, prev_py)
            bbox_padded = [
                max(0, round(bbox[0] - BOX_PADDING)),
                max(0, round(bbox[1] - BOX_PADDING)),
                min(img_w - 1, round(bbox[2] + BOX_PADDING)),
                min(img_h - 1, round(bbox[3] + BOX_PADDING)),
            ]

            # Annotate
            ann_img = annotate_arrowhead(png_img, tip_px, tip_py, prev_px, prev_py)
            out_img_path = ann_img_dir / f"{pid}_arrow_{arrow_idx}.png"
            ann_img.save(str(out_img_path), "PNG")

            # Ground truth
            src_loc = immediate_container(conn, "source") or "None"
            tgt_loc = immediate_container(conn, "target") or "None"
            cond = conn["condition"] or "None"

            # Per-process arrow record
            proc_arrows.append({
                "arrow_id": arrow_idx,
                "source_node": conn["source"],
                "source_location": src_loc,
                "condition": cond,
                "target_node": conn["target"],
                "target_location": tgt_loc,
                "matched_arrowhead_bbox": bbox_padded,
                "annotated_image_path": str(out_img_path.absolute()),
            })

            total_arrows += 1

        # Save per-process JSON
        proc_json = {
            "image_name": pid,
            "image_path": str(svg_path.absolute()),
            "arrows": proc_arrows,
        }
        proc_json_path = proc_out_dir / f"{pid}.json"
        with open(proc_json_path, "w", encoding="utf-8") as f:
            json.dump(proc_json, f, indent=2, ensure_ascii=False)

        processed += 1
        if processed % 20 == 0:
            print(f"  processed {processed} files, {total_arrows} arrows so far")

    return processed, skipped, total_arrows


def main():
    parser = argparse.ArgumentParser(description="Generate BPMN arrow intermediate data")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        help="Which splits to process")
    args = parser.parse_args()

    print("=" * 60)
    print("BPMN Arrow Intermediate Data Generator (SVG mode)")
    print(f"Splits : {args.splits}")
    print("=" * 60)

    total_processed = total_skipped = total_arrows = 0
    for split in args.splits:
        p, s, a = process_split(split)
        total_processed += p
        total_skipped   += s
        total_arrows    += a

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Processed : {total_processed} files")
    print(f"  Skipped   : {total_skipped} files")
    print(f"  Arrows    : {total_arrows}")
    print("=" * 60)


if __name__ == "__main__":
    main()
