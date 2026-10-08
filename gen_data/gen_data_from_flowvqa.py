#!/usr/bin/env python3
"""
Generate FlowVQA arrow-triplet training data, end to end.

Stage 1 — render: for each FlowVQA entry, render its mermaid code to PNG + SVG
          (via the `mmdc` Mermaid CLI) into a per-key folder.
Stage 2 — extract: parse the SVG to recover each arrow's (source_node, target_node,
          condition) and its arrowhead bbox, draw one annotated image per arrow
          (a blue box on the arrowhead), and write <key>.json alongside.

(This merges the former gen_data_from_flowvqa.py + gen_data_from_flowvqa2.py.)

Usage:
    python gen_data/gen_data_from_flowvqa.py
"""

import json
import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not found; annotated images will not be drawn.")

# ==================== Configuration ====================
JSON_PATH = "<DATA_ROOT>/flowvqa/Data/test_full.json"
OUTPUT_BASE_DIR = "<DATA_ROOT>/data_4_training/flowvqa_test"


# ==================== Stage 1: render mermaid -> PNG + SVG ====================

def render_mermaid_to_images(json_path: str, output_base_dir: str):
    """Render each entry's mermaid code to PNG + SVG under output_base_dir/<key>/."""
    if not os.path.exists(json_path):
        print(f"Error: file not found {json_path}")
        return []

    # Unified mermaid config: disable max-width so layout does not auto-scale,
    # use pure-SVG text (no HTML labels) so text is parseable from the SVG.
    mermaid_config = {
        "flowchart": {
            "htmlLabels": False,
            "nodePadding": 20,
            "useMaxWidth": False,
            "curve": "basis",
        },
        "theme": "default",
        "themeVariables": {
            "fontFamily": "arial, sans-serif",
            "fontSize": "16px",
        },
    }
    m_config_path = "temp_mermaid_config.json"
    with open(m_config_path, "w") as f:
        json.dump(mermaid_config, f)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rendered_keys = []
    for key, content in data.items():
        mermaid_code = content.get("mermaid")
        if not mermaid_code:
            continue

        target_dir = os.path.join(output_base_dir, key)
        os.makedirs(target_dir, exist_ok=True)

        mmd_file = os.path.join(target_dir, f"{key}.mmd")
        output_png = os.path.join(target_dir, f"{key}.png")
        output_svg = os.path.join(target_dir, f"{key}.svg")

        with open(mmd_file, "w", encoding="utf-8") as f_mmd:
            f_mmd.write(mermaid_code)

        try:
            common_args = ["mmdc", "-i", mmd_file, "-c", m_config_path, "-b", "white", "-s", "1"]
            subprocess.run(common_args + ["-o", output_png], check=True, capture_output=True)
            subprocess.run(common_args + ["-o", output_svg], check=True, capture_output=True)
            rendered_keys.append(key)
            print(f"Rendered: {key}")
        except subprocess.CalledProcessError as e:
            print(f"Render failed {key}: {e.stderr.decode()}")
        except Exception as e:
            print(f"Other error {key}: {e}")

    if os.path.exists(m_config_path):
        os.remove(m_config_path)
    return rendered_keys


# ==================== Geometry helpers ====================

def parse_svg_path_last_segment(d_str):
    """Extract the last two points of a path `d` string (for the tangent direction).
    Handles mixed M/L/C commands — we only need the last two coordinate pairs."""
    nums = [float(n) for n in re.findall(r"-?[\d\.]+", d_str)]
    if len(nums) >= 4:
        prev_pt = (nums[-4], nums[-3])   # second-to-last point (gives direction)
        end_pt = (nums[-2], nums[-1])    # endpoint (arrow tip)
        return prev_pt, end_pt
    elif len(nums) >= 2:
        return (0, 0), (nums[-2], nums[-1])
    return (0, 0), (0, 0)


def calculate_arrowhead_bbox(prev_pt, end_pt, arrow_len=12, arrow_width=10, padding=5):
    """Compute the arrowhead bounding box from the segment direction.
    Returns [min_x, min_y, max_x, max_y]."""
    dx = end_pt[0] - prev_pt[0]
    dy = end_pt[1] - prev_pt[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return [end_pt[0] - padding, end_pt[1] - padding, end_pt[0] + padding, end_pt[1] + padding]

    ux, uy = dx / length, dy / length     # unit vector
    vx, vy = -uy, ux                       # perpendicular vector

    p1 = end_pt                            # arrow tip
    base_cx = end_pt[0] - ux * arrow_len   # base center (back off along the segment)
    base_cy = end_pt[1] - uy * arrow_len
    half_w = arrow_width / 2
    p2 = (base_cx + vx * half_w, base_cy + vy * half_w)
    p3 = (base_cx - vx * half_w, base_cy - vy * half_w)

    xs = [p1[0], p2[0], p3[0]]
    ys = [p1[1], p2[1], p3[1]]
    return [min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding]


# ==================== SVG parsing ====================

def get_node_mapping(root, ns):
    """Map each node's code -> its text content."""
    node_map = {}
    for node in root.findall(".//g", ns):
        if "node" not in node.get("class", ""):
            continue
        node_id = node.get("id", "")
        match = re.search(r"flowchart-(.+?)-\d+$", node_id)
        if match:
            node_code = match.group(1)
            text_content = ""
            foreign_obj = node.find(".//foreignObject", ns)
            if foreign_obj is not None:
                html_content = ET.tostring(foreign_obj, encoding="unicode", method="text")
                text_content = " ".join(html_content.split()).strip()
            node_map[node_code] = text_content
    return node_map


def get_edge_labels(root, ns):
    """Map each edge id -> its label text (the condition)."""
    labels = {}
    for label_group in root.findall(".//g[@class='edgeLabel']", ns):
        inner_label = label_group.find(".//g[@class='label']", ns)
        if inner_label is not None:
            edge_id = inner_label.get("data-id")
            html_content = ET.tostring(inner_label, encoding="unicode", method="text")
            text = " ".join(html_content.split()).strip()
            if text:
                labels[edge_id] = text
    return labels


def extract_flowchart_with_bbox(svg_content, base_image_path_input=None, output_dir="annotated_images"):
    """Extract arrow info from the SVG and draw one annotated image per arrow."""
    # Strip namespaces
    svg_content = re.sub(r' xmlns="[^"]+"', "", svg_content, count=1)
    svg_content = re.sub(r' xmlns:xlink="[^"]+"', "", svg_content, count=1)
    root = ET.fromstring(svg_content)
    ns = {}

    node_map = get_node_mapping(root, ns)
    edge_labels = get_edge_labels(root, ns)

    arrows_data = {"arrows": []}
    arrow_id_counter = 0

    base_img = None
    should_draw = False
    if CV2_AVAILABLE and base_image_path_input and os.path.exists(base_image_path_input):
        base_img = cv2.imread(base_image_path_input)
        should_draw = base_img is not None
    if should_draw and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for path in root.findall(".//path", ns):
        if "flowchart-link" not in path.get("class", ""):
            continue

        edge_id = path.get("id", "")
        d_attr = path.get("d")

        id_match = re.search(r"^L_(.+?)_(.+?)_\d+$", edge_id)
        if not id_match:
            continue
        start_code, end_code = id_match.group(1), id_match.group(2)
        start_text = node_map.get(start_code, f"Unknown: {start_code}")
        end_text = node_map.get(end_code, f"Unknown: {end_code}")

        condition_text = edge_labels.get(edge_id, [])
        if isinstance(condition_text, str):
            condition_text = [condition_text]

        prev_pt, end_pt = parse_svg_path_last_segment(d_attr)
        arrow_bbox = calculate_arrowhead_bbox(prev_pt, end_pt, arrow_len=10, arrow_width=10, padding=5)

        # One annotated image per arrow (blue box on the arrowhead)
        output_path_str = ""
        if should_draw:
            try:
                current_img = base_img.copy()
                min_x, min_y, max_x, max_y = arrow_bbox
                cv2.rectangle(current_img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)
                full_path = os.path.join(output_dir, f"arrow_{arrow_id_counter}.png")
                cv2.imwrite(full_path, current_img)
                output_path_str = os.path.abspath(full_path)
            except Exception as e:
                print(f"Drawing error for arrow {arrow_id_counter}: {e}")

        arrows_data["arrows"].append({
            "arrow_id": arrow_id_counter,
            "source_node": [start_text],
            "target_node": [end_text],
            "condition": condition_text,
            "matched_arrowhead_bbox": arrow_bbox,
            "annotated_image_path": output_path_str,
        })
        arrow_id_counter += 1

    return arrows_data


def process_single_folder(folder_path):
    """Parse the SVG in one folder (e.g. code00000) and write its <name>.json."""
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    svg_files = list(folder_path.glob("*.svg"))
    png_files = list(folder_path.glob("*.png"))
    if not svg_files:
        print(f"⚠️  {folder_name}: no SVG file, skipping")
        return False
    if not png_files:
        print(f"⚠️  {folder_name}: no PNG file, skipping")
        return False

    svg_file, png_file = svg_files[0], png_files[0]
    annotated_dir = folder_path / "annotated_images"
    annotated_dir.mkdir(exist_ok=True)

    try:
        with open(svg_file, "r", encoding="utf-8") as f:
            svg_content = f.read()
    except Exception as e:
        print(f"❌ {folder_name}: cannot read SVG - {e}")
        return False

    try:
        results = extract_flowchart_with_bbox(
            svg_content, base_image_path_input=str(png_file), output_dir=str(annotated_dir)
        )
    except Exception as e:
        print(f"❌ {folder_name}: processing failed - {e}")
        return False

    json_file = folder_path / f"{folder_name}.json"
    try:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"✅ {folder_name}: {len(results['arrows'])} arrows")
        return True
    except Exception as e:
        print(f"❌ {folder_name}: cannot save JSON - {e}")
        return False


def main():
    # Stage 1: render every mermaid diagram to PNG + SVG
    render_mermaid_to_images(JSON_PATH, OUTPUT_BASE_DIR)

    # Stage 2: extract arrows from each rendered folder
    root_path = Path(OUTPUT_BASE_DIR)
    if not root_path.exists():
        print(f"❌ Error: directory not found - {OUTPUT_BASE_DIR}")
        return
    subfolders = sorted([f for f in root_path.iterdir() if f.is_dir()])
    if not subfolders:
        print("⚠️  no subfolders found")
        return

    print(f"\n📁 extracting arrows from {len(subfolders)} folders...\n")
    success_count = fail_count = 0
    for folder in subfolders:
        if process_single_folder(folder):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 50}")
    print("Done!")
    print(f"✅ success: {success_count}")
    print(f"❌ failed:  {fail_count}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
