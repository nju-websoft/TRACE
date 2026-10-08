#!/usr/bin/env python3
"""
Generate FlowGen training data from source diagram files.

Flow:
  1. Convert .dot/.mmd/.puml to SVG
  2. Render SVG to PNG (reduced size)
     - dot/graphviz/plantuml: cairosvg
     - mermaid: mmdc (foreignObject not supported by cairosvg)
  3. Parse SVG for arrow coordinates
  4. Annotate arrowheads with blue boxes
  5. Write one per-sample JSON of arrow triplets per source diagram

Usage:
    python gen_data_from_flowgen.py [--types diagrams graphviz mermaid plantuml]
                                    [--difficulties easy medium hard]
"""

import argparse
import io
import json
import math
import os
import random
import re
import subprocess
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw

# ── Paths & Config ────────────────────────────────────────────────────────────
FLOWGEN_ROOT = Path("<DATA_ROOT>/flowgen")
OUTPUT_ROOT  = Path("<DATA_ROOT>/data_4_training/flowgen")
PLANTUML_JAR = Path("/tmp/plantuml.jar")

MAX_DIM     = 2000
BOX_WIDTH   = 3
BOX_PADDING = 5
RANDOM_SEED = 42
DEV_RATIO   = 0.1

SRC_EXT = {
    "diagrams": ".dot",
    "graphviz": ".dot",
    "mermaid":  ".mmd",
    "plantuml": ".puml",
}

# ══════════════════════════════════════════════════════════════════════════════
#  Source-file parsers  (node names & edge labels)
# ══════════════════════════════════════════════════════════════════════════════

def parse_dot_nodes(path: Path) -> dict[str, str]:
    """Parse .dot → {node_id: label}. Handles both quoted and unquoted labels."""
    nodes = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r'\s*(\w+)\s*\[', line)
            if not m:
                continue
            nid = m.group(1)
            # Try quoted label first, then unquoted
            lm = re.search(r'label="([^"]*)"', line)
            if not lm:
                lm = re.search(r'\blabel=(\w+)', line)
            if lm:
                nodes[nid] = lm.group(1).strip()
    return nodes


def parse_dot_edge_labels(path: Path) -> dict[tuple[str, str], str]:
    """Parse .dot → {(src_id, tgt_id): label}."""
    edges = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r'\s*(\w+)\s*->\s*(\w+)\s*\[(.*?)\]', line)
            if m:
                src, tgt, attrs = m.group(1), m.group(2), m.group(3)
                lm = re.search(r'label="([^"]*)"', attrs)
                if not lm:
                    lm = re.search(r'\blabel=(\w+)', attrs)
                label = lm.group(1).strip() if lm else ""
                edges[(src, tgt)] = label
    return edges


def parse_mmd_nodes(path: Path) -> dict[str, str]:
    """Parse .mmd → {node_id: name}."""
    nodes = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r'(\w+)\s*([\[\({>]+)(.+)', line)
            if m:
                nid = m.group(1)
                # Skip keywords
                if nid in ("subgraph", "end", "graph", "style", "classDef",
                            "linkStyle", "click"):
                    continue
                rest = m.group(3)
                name = re.sub(r'[\]\)}>]+\s*$', '', rest).strip()
                # Strip mermaid shape delimiters: /text/, \text\, etc.
                name = re.sub(r'^[/\\]+|[/\\]+$', '', name).strip()
                # Virtual nodes (empty or space-only) → keep with empty label
                # so build_vnode_resolver can resolve them
                if name.strip() in ("", " "):
                    nodes[nid] = ""
                else:
                    nodes[nid] = name
    return nodes


def parse_puml_nodes(path: Path) -> dict[str, str]:
    """Parse .puml → {node_id: name}."""
    nodes = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r'"([^"]+)"\s+as\s+(\w+)', line)
            if m:
                name = m.group(1).strip()
                nid = m.group(2)
                # Skip group containers (Group0, etc.)
                if nid.startswith("Group"):
                    continue
                # Virtual nodes (empty or space-only) → keep with empty label
                if not name or name == " ":
                    nodes[nid] = ""
                else:
                    nodes[nid] = name
    return nodes


# ══════════════════════════════════════════════════════════════════════════════
#  Containment parsers  (node_id → cluster/subgraph label)
# ══════════════════════════════════════════════════════════════════════════════

def parse_dot_containment(path: Path) -> dict[str, str]:
    """Parse .dot subgraph clusters → {node_id: cluster_label}.
    Works for both graphviz and diagrams .dot files.
    Handles: subgraph cluster_Label { ... nestX_Y [...] ... }
    For graphviz, cluster label comes from label= attribute inside subgraph.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    containment: dict[str, str] = {}
    for m in re.finditer(
        r'subgraph\s+cluster_(\S+(?:\s+\S+)*)\s*\{([^}]+)\}',
        content,
    ):
        sg_key = m.group(1).strip()
        sg_body = m.group(2)
        # Try to get the display label from label= attribute
        label_m = re.search(r'label="([^"]*)"', sg_body)
        sg_label = label_m.group(1) if label_m else sg_key
        # Find all nested node IDs
        for nid in re.findall(r'(\w+)\s*\[', sg_body):
            containment[nid] = sg_label
    return containment


def parse_mmd_containment(path: Path) -> dict[str, str]:
    """Parse .mmd subgraph blocks → {node_id: subgraph_name}."""
    containment: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        content = f.read()

    for m in re.finditer(
        r'subgraph\s+(.+?)\n(.*?)end',
        content, re.DOTALL,
    ):
        sg_name = m.group(1).strip()
        sg_body = m.group(2)
        for nid_m in re.finditer(r'(\w+)[\[\({]', sg_body):
            containment[nid_m.group(1)] = sg_name
    return containment


def parse_puml_containment(path: Path) -> dict[str, str]:
    """Parse .puml rectangle/package groups → {node_id: group_label}."""
    containment: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Match: rectangle "Label" as GroupX ... { ... }
    for m in re.finditer(
        r'(?:rectangle|package|frame)\s+"([^"]+)"\s+as\s+\w+[^{]*\{([^}]+)\}',
        content, re.DOTALL,
    ):
        group_label = m.group(1)
        group_body = m.group(2)
        for nid_m in re.finditer(r'\bas\s+(\w+)', group_body):
            containment[nid_m.group(1)] = group_label
    return containment


def get_containment(src_path: Path, dtype: str) -> dict[str, str]:
    """Get node containment mapping for any diagram type."""
    if dtype in ("diagrams", "graphviz"):
        return parse_dot_containment(src_path)
    elif dtype == "mermaid":
        return parse_mmd_containment(src_path)
    elif dtype == "plantuml":
        return parse_puml_containment(src_path)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  Virtual node resolution
# ══════════════════════════════════════════════════════════════════════════════

def build_vnode_resolver(
    nodes: dict[str, str],
    edges: list[tuple[str, str, str]],
) -> dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
    """
    Build virtual node resolver.

    Args:
        nodes: {node_id: label} — virtual nodes have empty label
        edges: [(src_id, tgt_id, label), ...]

    Returns:
        {vnode_id: ([(real_pred_id, condition)], [(real_succ_id, condition)])}
        where predecessors/successors are resolved transitively through
        chains of virtual nodes, carrying the edge condition closest to
        the virtual node being resolved.
    """
    vnodes = {nid for nid, label in nodes.items() if not label}
    if not vnodes:
        return {}

    # Build adjacency with labels
    fwd: dict[str, list[tuple[str, str]]] = {}   # node → [(succ, label)]
    bwd: dict[str, list[tuple[str, str]]] = {}   # node → [(pred, label)]
    for s, t, lbl in edges:
        fwd.setdefault(s, []).append((t, lbl))
        bwd.setdefault(t, []).append((s, lbl))

    def _real_successors(nid: str, visited: set | None = None) -> list[tuple[str, str]]:
        if visited is None:
            visited = set()
        if nid in visited:
            return []
        visited.add(nid)
        result = []
        for s, lbl in fwd.get(nid, []):
            if s in vnodes:
                # Propagate: use this edge's label if set, else inherit from deeper
                for deep_id, deep_lbl in _real_successors(s, visited):
                    result.append((deep_id, lbl if lbl else deep_lbl))
            else:
                result.append((s, lbl))
        return result

    def _real_predecessors(nid: str, visited: set | None = None) -> list[tuple[str, str]]:
        if visited is None:
            visited = set()
        if nid in visited:
            return []
        visited.add(nid)
        result = []
        for p, lbl in bwd.get(nid, []):
            if p in vnodes:
                for deep_id, deep_lbl in _real_predecessors(p, visited):
                    result.append((deep_id, lbl if lbl else deep_lbl))
            else:
                result.append((p, lbl))
        return result

    resolver = {}
    for v in vnodes:
        resolver[v] = (_real_predecessors(v), _real_successors(v))
    return resolver


# ══════════════════════════════════════════════════════════════════════════════
#  SVG conversion
# ══════════════════════════════════════════════════════════════════════════════

# ── Diagrams rendering (the `diagrams` library renders DOT -> SVG with shape icons) ──
_DIAGRAMS_SHAPE_MAP = None


def _get_diagrams_shape_map():
    """Lazily import the `diagrams` library and return the shape map."""
    global _DIAGRAMS_SHAPE_MAP
    if _DIAGRAMS_SHAPE_MAP is not None:
        return _DIAGRAMS_SHAPE_MAP
    from diagrams.programming.flowchart import (
        InputOutput, Action, PredefinedProcess, Database,
        MultipleDocuments, Preparation, Document, Decision,
        Display, Inspection, InternalStorage, LoopLimit,
        ManualLoop, ManualInput, StartEnd,
    )
    _DIAGRAMS_SHAPE_MAP = {
        "inputoutput": InputOutput, "action": Action,
        "predefinedprocess": PredefinedProcess, "database": Database,
        "multipledocuments": MultipleDocuments, "preparation": Preparation,
        "document": Document, "box": Decision, "rect": Display,
        "rectangle": InternalStorage, "ellipse": LoopLimit,
        "diamond": ManualLoop, "hexagon": ManualInput,
        "parallelogram": StartEnd, "circle": Inspection,
        "vnode_shape": Inspection,
    }
    return _DIAGRAMS_SHAPE_MAP


def _parse_dot_for_diagrams(dot_path: Path):
    """Parse a .dot file: rankdir, nodes, edges, virtual nodes, and subgraphs."""
    with open(dot_path, encoding="utf-8") as f:
        content = f.read()

    rankdir_m = re.search(r'rankdir=(\w+)', content)
    direction = rankdir_m.group(1) if rankdir_m else "TB"

    nodes = {}
    for m in re.finditer(
        r'(\w+)\s*\[label="([^"]*)".*?color="([^"]*)".*?fillcolor="([^"]*)".*?shape=(\w+)',
        content,
    ):
        nid, label, color, fillcolor, shape = m.groups()
        # virtual node: empty label + circle + fixedsize
        is_vnode = (label == "" and shape == "circle" and "fixedsize=true" in content.split(nid)[1].split(']')[0])
        nodes[nid] = {
            "label": label, "color": color,
            "fillcolor": fillcolor, "shape": shape,
            "is_vnode": is_vnode,
        }

    subgraphs = {}
    for m in re.finditer(
        r'subgraph\s+cluster_([^\s{]+(?:\s+[^\s{]+)*)\s*\{([^}]+)\}',
        content,
    ):
        sg_label = m.group(1).strip()
        sg_body = m.group(2)
        sg_node_ids = re.findall(r'(nest\w+)\s*\[', sg_body)
        subgraphs[sg_label] = sg_node_ids

    node_to_subgraph = {}
    for sg_label, sg_nodes in subgraphs.items():
        for nid in sg_nodes:
            node_to_subgraph[nid] = sg_label

    edges = []
    for m in re.finditer(r'(\w+)\s*->\s*(\w+)\s*\[(.+?)\]', content):
        src, tgt, attrs = m.groups()
        label_m = re.search(r'label="([^"]*)"', attrs)
        color_m = re.search(r'color="([^"]*)"', attrs)
        style_m = re.search(r'style=(\w+)', attrs)
        pw_m = re.search(r'penwidth=([\d.]+)', attrs)
        edges.append({
            "src": src, "tgt": tgt,
            "label": label_m.group(1) if label_m else "",
            "color": color_m.group(1) if color_m else "#000000",
            "style": style_m.group(1) if style_m else "solid",
            "penwidth": pw_m.group(1) if pw_m else "1.5",
        })

    return direction, nodes, edges, subgraphs, node_to_subgraph


def _embed_svg_images(svg_content: str) -> str:
    """Replace local <image> file paths with base64 data URIs (cairosvg-compatible)."""
    import base64
    _icon_cache: dict = {}

    def _embed(m):
        tag = m.group(0)
        href_m = re.search(r'xlink:href="([^"]+)"', tag)
        if not href_m:
            return tag
        path = href_m.group(1)
        if path.startswith("data:"):
            return tag
        if path not in _icon_cache:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                _icon_cache[path] = f"data:image/png;base64,{b64}"
            except Exception:
                return tag
        return tag.replace(f'xlink:href="{path}"',
                           f'xlink:href="{_icon_cache[path]}"')

    return re.sub(r'<image[^>]*/?\s*>', _embed, svg_content)


def diagrams_dot_to_svg(src_path: Path, svg_path: Path) -> bool:
    """Render a .dot file to SVG via the `diagrams` library (labelloc=c centers text)."""
    try:
        import os
        from diagrams import Diagram, Edge, Cluster

        shape_map = _get_diagrams_shape_map()
        direction, nodes, edges, subgraphs, node_to_subgraph = _parse_dot_for_diagrams(src_path)

        out_dir = str(svg_path.parent.resolve())
        out_stem = svg_path.stem
        out_filename = os.path.join(out_dir, out_stem)

        graph_attrs = {"splines": "polyline", "fontsize": "10", "dpi": "120", "rankdir": direction}
        node_attr = {"style": "solid", "fontsize": "9", "labelloc": "c"}

        original_cwd = os.getcwd()
        try:
            with Diagram(
                outformat="svg", filename=out_filename,
                show=False, graph_attr=graph_attrs,
                node_attr=node_attr, direction=direction,
            ):
                node_refs = {}
                for nid, info in nodes.items():
                    if nid in node_to_subgraph:
                        continue
                    if info["is_vnode"]:
                        vnode_cls = shape_map["vnode_shape"]
                        node_refs[nid] = vnode_cls(
                            "", shape="point", fixedsize="true",
                            height="0.1", width="0.1", style="dashed",
                            color="#888888", fillcolor="#FFFFFF",
                        )
                    else:
                        shape_cls = shape_map.get(info["shape"], shape_map["action"])
                        node_refs[nid] = shape_cls(
                            info["label"], fillcolor=info["fillcolor"], color=info["color"],
                        )

                for sg_label, sg_node_ids in subgraphs.items():
                    with Cluster(sg_label):
                        for nid in sg_node_ids:
                            if nid not in nodes:
                                continue
                            info = nodes[nid]
                            shape_cls = shape_map.get(info["shape"], shape_map["action"])
                            node_refs[nid] = shape_cls(
                                info["label"], fillcolor=info["fillcolor"], color=info["color"],
                            )

                for e in edges:
                    if e["src"] in node_refs and e["tgt"] in node_refs:
                        edge_attrs = {"color": e["color"], "style": e["style"], "penwidth": e["penwidth"]}
                        if e["label"]:
                            edge_attrs["label"] = e["label"]
                        node_refs[e["src"]] >> Edge(**edge_attrs) >> node_refs[e["tgt"]]
        finally:
            os.chdir(original_cwd)

        if not (svg_path.exists() and svg_path.stat().st_size > 0):
            return False

        # Embed icon images as base64 (cairosvg cannot follow local file refs)
        with open(svg_path, encoding="utf-8") as f:
            svg_content = f.read()
        svg_content = _embed_svg_images(svg_content)
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return True
    except Exception as e:
        print(f"      diagrams svg error: {e}")
        return False


def convert_to_svg(src_path: Path, svg_path: Path, dtype: str) -> bool:
    """Convert a source file to SVG. Returns True on success.

    `diagrams` is rendered via the diagrams library (embeds shape icons); the other
    types use dot / mmdc / plantuml directly.
    """
    if svg_path.exists():
        return True
    try:
        if dtype == "diagrams":
            return diagrams_dot_to_svg(src_path, svg_path)
        if dtype == "graphviz":
            subprocess.run(
                ["dot", "-Tsvg", str(src_path), "-o", str(svg_path)],
                capture_output=True, timeout=30,
            )
        elif dtype == "mermaid":
            subprocess.run(
                ["mmdc", "-i", str(src_path), "-o", str(svg_path), "-e", "svg"],
                capture_output=True, timeout=60,
            )
        elif dtype == "plantuml":
            subprocess.run(
                ["java", "-jar", str(PLANTUML_JAR), "-tsvg",
                 str(src_path), "-o", str(svg_path.parent)],
                capture_output=True, timeout=60,
            )
        return svg_path.exists()
    except Exception as e:
        print(f"    Convert error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  SVG arrow parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_viewbox(content: str) -> tuple[float, float, float, float]:
    m = re.search(r'viewBox="([^"]+)"', content)
    if m:
        parts = [float(x) for x in re.split(r'[,\s]+', m.group(1).strip())]
        if len(parts) >= 4:
            return tuple(parts[:4])
    return (0.0, 0.0, 100.0, 100.0)


def _parse_polygon_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG polygon points attribute → [(x,y), ...]."""
    pts = []
    for tok in points_str.strip().split():
        coords = tok.split(",")
        if len(coords) == 2:
            try:
                pts.append((float(coords[0]), float(coords[1])))
            except ValueError:
                pass
    if not pts:
        # flat list: x0,y0,x1,y1,...
        nums = re.findall(r'[-+]?\d*\.?\d+', points_str)
        nums = [float(n) for n in nums]
        for i in range(0, len(nums) - 1, 2):
            pts.append((nums[i], nums[i + 1]))
    return pts


def parse_dot_svg_arrows(svg_path: Path, src_path: Path):
    """
    Parse graphviz SVG + source .dot for arrows.
    Returns: (arrows_list, viewbox)
      arrows_list: [{src_name, tgt_name, label, tip_vx, tip_vy, prev_vx, prev_vy}, ...]
      Coordinates are in viewport space.
    """
    with open(svg_path, encoding="utf-8") as f:
        content = f.read()

    vb = _parse_viewbox(content)

    # Get transform from graph0 group
    sx, sy, tx, ty = 1.0, 1.0, 0.0, 0.0
    tm = re.search(r'<g\s+id="graph0"[^>]*transform="([^"]+)"', content)
    if tm:
        t_str = tm.group(1)
        sm = re.search(r'scale\(([\d.e+-]+)\s+([\d.e+-]+)\)', t_str)
        if sm:
            sx, sy = float(sm.group(1)), float(sm.group(2))
        trm = re.search(r'translate\(([\d.e+-]+)\s+([\d.e+-]+)\)', t_str)
        if trm:
            tx, ty = float(trm.group(1)), float(trm.group(2))

    nodes = parse_dot_nodes(src_path)
    edge_labels = parse_dot_edge_labels(src_path)

    arrows = []
    edge_re = re.compile(
        r'<g\s+id="edge\d+"[^>]*class="edge"[^>]*>(.*?)</g>',
        re.DOTALL,
    )

    for em in edge_re.finditer(content):
        block = em.group(1)

        # Edge title → source/target IDs
        title_m = re.search(r'<title>([^<]+)</title>', block)
        if not title_m:
            continue
        title = title_m.group(1).replace("&#45;", "-").replace("&gt;", ">")
        parts = title.split("->")
        if len(parts) != 2:
            continue
        src_id, tgt_id = parts[0].strip(), parts[1].strip()

        src_name = nodes.get(src_id, "")
        tgt_name = nodes.get(tgt_id, "")

        # Arrowhead polygon
        poly_m = re.search(r'<polygon[^>]*points="([^"]+)"', block)
        if not poly_m:
            continue
        pts = _parse_polygon_points(poly_m.group(1))
        if len(pts) < 3:
            continue

        tip = pts[0]
        base_pts = [p for p in pts[1:]
                     if abs(p[0] - tip[0]) > 0.1 or abs(p[1] - tip[1]) > 0.1]
        if not base_pts:
            base_pts = pts[1:2]
        base = (sum(p[0] for p in base_pts) / len(base_pts),
                sum(p[1] for p in base_pts) / len(base_pts))

        # Local → viewport (SVG transform: scale then translate applied R-to-L)
        tip_vx = (tip[0] + tx) * sx
        tip_vy = (tip[1] + ty) * sy
        prev_vx = (base[0] + tx) * sx
        prev_vy = (base[1] + ty) * sy

        label = edge_labels.get((src_id, tgt_id), "")

        # Edge label from SVG text (fallback)
        if not label:
            text_m = re.search(r'<text[^>]*>([^<]+)</text>', block)
            if text_m:
                label = text_m.group(1).strip()

        arrows.append({
            "src_id": src_id, "tgt_id": tgt_id,
            "src_name": src_name, "tgt_name": tgt_name, "label": label,
            "tip_vx": tip_vx, "tip_vy": tip_vy,
            "prev_vx": prev_vx, "prev_vy": prev_vy,
        })

    return arrows, vb


def parse_mermaid_svg_arrows(svg_path: Path, src_path: Path):
    """
    Parse mermaid SVG + source .mmd for arrows.
    Returns: (arrows_list, viewbox)
    """
    with open(svg_path, encoding="utf-8") as f:
        content = f.read()

    vb = _parse_viewbox(content)
    nodes = parse_mmd_nodes(src_path)

    # ---- edge paths ----
    raw_arrows = {}
    for pm in re.finditer(r'<path\s([^>]+?)/?>', content):
        attrs = pm.group(1)
        _NODE_ID = r'(?:(?:N|S|M)\d+|nest\d+_\d+)'
        id_m = re.search(
            rf'id="(L_({_NODE_ID})_({_NODE_ID})_(\d+))"', attrs,
        )
        if not id_m:
            continue
        edge_id = id_m.group(1)
        src_id, tgt_id, idx = id_m.group(2), id_m.group(3), id_m.group(4)

        d_m = re.search(r'\bd="(M[^"]+)"', attrs)
        if not d_m:
            # try matching d= that starts right after <path
            d_m = re.search(r'^d="(M[^"]+)"', attrs)
        if not d_m:
            continue

        nums = re.findall(r'[-+]?\d*\.?\d+', d_m.group(1))
        nums = [float(n) for n in nums]
        if len(nums) < 4:
            continue

        path_end_x, path_end_y = nums[-2], nums[-1]
        path_prev_x, path_prev_y = nums[-4], nums[-3]

        # Mermaid marker-end offset: the SVG marker refX=5 in a
        # viewBox 0..10, markerWidth=8 → the visible arrowhead tip
        # is (10-5)*(8/10)=4 SVG units AHEAD of the path endpoint.
        dx = path_end_x - path_prev_x
        dy = path_end_y - path_prev_y
        seg_len = math.hypot(dx, dy)
        if seg_len > 0:
            marker_offset = 4.0
            tip_x = path_end_x + dx / seg_len * marker_offset
            tip_y = path_end_y + dy / seg_len * marker_offset
        else:
            tip_x, tip_y = path_end_x, path_end_y
        prev_x, prev_y = path_prev_x, path_prev_y

        key = (src_id, tgt_id, idx)
        raw_arrows[key] = (tip_x, tip_y, prev_x, prev_y)

    # ---- edge labels ----
    edge_labels: dict[str, str] = {}
    _NID = r'(?:(?:N|S|M)\d+|nest\d+_\d+)'
    for lm in re.finditer(
        rf'class="label"\s+data-id="(L_{_NID}_{_NID}_\d+)"[^>]*>(.*?)</g>',
        content, re.DOTALL,
    ):
        eid = lm.group(1)
        block = lm.group(2)
        pm = re.search(r'<p>([^<]*)</p>', block)
        if pm:
            txt = pm.group(1).strip()
            if txt and txt not in ("&nbsp;", " "):
                edge_labels[eid] = txt

    # ---- assemble ----
    arrows = []
    for (src_id, tgt_id, idx), (tip_x, tip_y, prev_x, prev_y) in raw_arrows.items():
        src_name = nodes.get(src_id, "")
        tgt_name = nodes.get(tgt_id, "")

        edge_id = f"L_{src_id}_{tgt_id}_{idx}"
        label = edge_labels.get(edge_id, "")

        arrows.append({
            "src_id": src_id, "tgt_id": tgt_id,
            "src_name": src_name, "tgt_name": tgt_name, "label": label,
            "tip_vx": tip_x, "tip_vy": tip_y,
            "prev_vx": prev_x, "prev_vy": prev_y,
        })

    return arrows, vb


def parse_puml_svg_arrows(svg_path: Path, src_path: Path):
    """
    Parse plantuml SVG + source .puml for arrows.
    Returns: (arrows_list, viewbox)
    """
    with open(svg_path, encoding="utf-8") as f:
        content = f.read()

    vb = _parse_viewbox(content)
    nodes = parse_puml_nodes(src_path)

    arrows = []
    _NID = r'(?:(?:N|S|M)\d+|nest\d+_\d+)'
    link_re = re.compile(
        rf'<g\s+id="link_({_NID})_({_NID})"[^>]*>(.*?)</g>', re.DOTALL,
    )

    for lm in link_re.finditer(content):
        src_id, tgt_id = lm.group(1), lm.group(2)
        block = lm.group(3)

        src_name = nodes.get(src_id, "")
        tgt_name = nodes.get(tgt_id, "")

        # Arrowhead polygon
        poly_m = re.search(r'<polygon[^>]*points="([^"]+)"', block)
        if not poly_m:
            continue
        pts = _parse_polygon_points(poly_m.group(1))
        if len(pts) < 3:
            continue

        tip = pts[0]
        base_pts = [p for p in pts[1:]
                     if abs(p[0] - tip[0]) > 0.1 or abs(p[1] - tip[1]) > 0.1]
        if not base_pts:
            base_pts = pts[1:2]
        base = (sum(p[0] for p in base_pts) / len(base_pts),
                sum(p[1] for p in base_pts) / len(base_pts))

        # Edge label (text in link group)
        label = ""
        text_m = re.search(r'<text[^>]*>([^<]+)</text>', block)
        if text_m:
            label = text_m.group(1).strip()

        arrows.append({
            "src_id": src_id, "tgt_id": tgt_id,
            "src_name": src_name, "tgt_name": tgt_name, "label": label,
            "tip_vx": tip[0], "tip_vy": tip[1],
            "prev_vx": base[0], "prev_vy": base[1],
        })

    return arrows, vb


def parse_diagrams_svg_arrows(svg_path: Path, src_path: Path):
    """
    Parse diagrams-library SVG (UUID-based node IDs) + source .dot for arrows.
    Returns: (arrows_list, viewbox)
    """
    with open(svg_path, encoding="utf-8") as f:
        content = f.read()

    vb = _parse_viewbox(content)

    # Get transform from graph0 group
    sx, sy, tx, ty = 1.0, 1.0, 0.0, 0.0
    tm = re.search(r'<g\s+id="graph0"[^>]*transform="([^"]+)"', content)
    if tm:
        t_str = tm.group(1)
        sm = re.search(r'scale\(([\d.e+-]+)\s+([\d.e+-]+)\)', t_str)
        if sm:
            sx, sy = float(sm.group(1)), float(sm.group(2))
        trm = re.search(r'translate\(([\d.e+-]+)\s+([\d.e+-]+)\)', t_str)
        if trm:
            tx, ty = float(trm.group(1)), float(trm.group(2))

    # Build UUID → label mapping from SVG node groups
    uuid_to_label = {}
    node_re = re.compile(
        r'<g\s+id="node\d+"[^>]*class="node"[^>]*>(.*?)</g>', re.DOTALL,
    )
    for nm in node_re.finditer(content):
        block = nm.group(1)
        title_m = re.search(r'<title>([^<]+)</title>', block)
        if not title_m:
            continue
        uuid = title_m.group(1).strip()
        text_m = re.search(r'<text[^>]*>([^<]*)</text>', block)
        label = text_m.group(1).strip() if text_m else ""
        uuid_to_label[uuid] = label

    # Also need edge labels from .dot source (UUID edges don't carry labels)
    edge_labels = parse_dot_edge_labels(src_path)
    dot_nodes = parse_dot_nodes(src_path)

    # Build label → dot_node_id mapping for resolving UUIDs back to dot IDs
    label_to_dot_ids: dict[str, list[str]] = {}
    for nid, lbl in dot_nodes.items():
        label_to_dot_ids.setdefault(lbl, []).append(nid)

    # Build UUID → dot_node_id mapping (for virtual node resolution)
    # Match by label; for virtual nodes (empty label), match by order of appearance
    uuid_to_dot_id: dict[str, str] = {}
    used_dot_ids: set[str] = set()
    empty_dot_ids = [nid for nid, lbl in dot_nodes.items() if not lbl]
    empty_idx = 0
    for uuid, label in uuid_to_label.items():
        if label:
            candidates = label_to_dot_ids.get(label, [])
            for c in candidates:
                if c not in used_dot_ids:
                    uuid_to_dot_id[uuid] = c
                    used_dot_ids.add(c)
                    break
        else:
            if empty_idx < len(empty_dot_ids):
                uuid_to_dot_id[uuid] = empty_dot_ids[empty_idx]
                empty_idx += 1

    arrows = []
    edge_re = re.compile(
        r'<g\s+id="edge\d+"[^>]*class="edge"[^>]*>(.*?)</g>', re.DOTALL,
    )

    for em in edge_re.finditer(content):
        block = em.group(1)

        # Edge title: UUID1->UUID2
        title_m = re.search(r'<title>([^<]+)</title>', block)
        if not title_m:
            continue
        title = title_m.group(1).replace("&#45;", "-").replace("&gt;", ">")
        parts = title.split("->")
        if len(parts) != 2:
            continue
        src_uuid, tgt_uuid = parts[0].strip(), parts[1].strip()

        src_name = uuid_to_label.get(src_uuid, "")
        tgt_name = uuid_to_label.get(tgt_uuid, "")
        src_dot_id = uuid_to_dot_id.get(src_uuid, src_uuid)
        tgt_dot_id = uuid_to_dot_id.get(tgt_uuid, tgt_uuid)

        # Arrowhead polygon
        poly_m = re.search(r'<polygon[^>]*points="([^"]+)"', block)
        if not poly_m:
            continue
        pts = _parse_polygon_points(poly_m.group(1))
        if len(pts) < 3:
            continue

        tip = pts[0]
        base_pts = [p for p in pts[1:]
                     if abs(p[0] - tip[0]) > 0.1 or abs(p[1] - tip[1]) > 0.1]
        if not base_pts:
            base_pts = pts[1:2]
        base = (sum(p[0] for p in base_pts) / len(base_pts),
                sum(p[1] for p in base_pts) / len(base_pts))

        # Local → viewport (SVG transform: scale then translate applied R-to-L)
        tip_vx = (tip[0] + tx) * sx
        tip_vy = (tip[1] + ty) * sy
        prev_vx = (base[0] + tx) * sx
        prev_vy = (base[1] + ty) * sy

        # Find edge label
        label = ""
        text_m = re.search(r'<text[^>]*>([^<]+)</text>', block)
        if text_m:
            label = text_m.group(1).strip()

        if not label:
            label = edge_labels.get((src_dot_id, tgt_dot_id), "")

        arrows.append({
            "src_id": src_dot_id, "tgt_id": tgt_dot_id,
            "src_name": src_name, "tgt_name": tgt_name, "label": label,
            "tip_vx": tip_vx, "tip_vy": tip_vy,
            "prev_vx": prev_vx, "prev_vy": prev_vy,
        })

    return arrows, vb


# ══════════════════════════════════════════════════════════════════════════════
#  SVG / source → PNG
# ══════════════════════════════════════════════════════════════════════════════

def svg_to_png_cairosvg(svg_path: Path, vb: tuple) -> tuple[Image.Image, float]:
    """
    Render SVG → PNG via cairosvg.  Good for dot / plantuml.
    Returns (image, scale).  scale maps viewport coords → pixel coords.
    """
    vb_x, vb_y, vb_w, vb_h = vb
    scale = min(1.0, MAX_DIM / max(vb_w, vb_h)) if max(vb_w, vb_h) > 0 else 1.0
    out_w = max(1, int(vb_w * scale))
    out_h = max(1, int(vb_h * scale))

    with open(svg_path, "rb") as f:
        svg_data = f.read()

    png_data = cairosvg.svg2png(bytestring=svg_data,
                                output_width=out_w, output_height=out_h)
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img)
    return bg.convert("RGB"), scale


def mermaid_to_png(src_path: Path, png_path: Path, vb: tuple) -> tuple[Image.Image, float, float]:
    """
    Render .mmd → PNG via mmdc.
    Returns (image, scale_x, scale_y).
    scale_x/y map viewport coords → pixel coords.
    """
    vb_x, vb_y, vb_w, vb_h = vb

    # Choose width so that max dim ≤ MAX_DIM
    target_w = int(vb_w)
    target_h = int(vb_h)
    overall_scale = min(1.0, MAX_DIM / max(target_w, target_h))
    render_w = max(1, int(target_w * overall_scale))

    if not png_path.exists():
        subprocess.run(
            ["mmdc", "-i", str(src_path), "-o", str(png_path),
             "-w", str(render_w)],
            capture_output=True, timeout=60,
        )

    if not png_path.exists():
        raise FileNotFoundError(f"mmdc failed to create {png_path}")

    img = Image.open(png_path).convert("RGB")
    img_w, img_h = img.size

    scale_x = img_w / vb_w if vb_w > 0 else 1.0
    scale_y = img_h / vb_h if vb_h > 0 else 1.0
    return img, scale_x, scale_y


# ══════════════════════════════════════════════════════════════════════════════
#  Annotation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _arrowhead_bbox(tip_x, tip_y, prev_x, prev_y):
    dx, dy = tip_x - prev_x, tip_y - prev_y
    length = math.hypot(dx, dy)
    if length == 0:
        return (tip_x - 8, tip_y - 8, tip_x + 8, tip_y + 8)
    angle = math.atan2(dy, dx)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    xs, ys = [], []
    for lx, ly in [(0, 0), (-10, -5), (-10, 5)]:
        rx = lx * cos_a - ly * sin_a + tip_x
        ry = lx * sin_a + ly * cos_a + tip_y
        xs.append(rx); ys.append(ry)
    return (min(xs), min(ys), max(xs), max(ys))


def annotate_arrowhead(img: Image.Image, tip_x, tip_y, prev_x, prev_y) -> Image.Image:
    x1, y1, x2, y2 = _arrowhead_bbox(tip_x, tip_y, prev_x, prev_y)
    w, h = img.size
    x1 = max(0, round(x1 - BOX_PADDING))
    y1 = max(0, round(y1 - BOX_PADDING))
    x2 = min(w - 1, round(x2 + BOX_PADDING))
    y2 = min(h - 1, round(y2 + BOX_PADDING))
    out = img.copy()
    ImageDraw.Draw(out).rectangle([(x1, y1), (x2, y2)],
                                  outline="blue", width=BOX_WIDTH)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Process one sample
# ══════════════════════════════════════════════════════════════════════════════

def process_sample(
    src_path: Path,
    svg_dir: Path,
    png_dir: Path,
    dtype: str,
    sample_id: str,
) -> list[dict] | None:
    """Process a single source file → annotated images + training records."""

    svg_path = svg_dir / f"{src_path.stem}.svg"

    # 1. Render the source file to SVG if not already present
    if not svg_path.exists():
        if not convert_to_svg(src_path, svg_path, dtype):
            return None

    # 2. Parse containment & build label→location
    containment = get_containment(src_path, dtype)
    label_to_location: dict[str, str] = {}

    # Get all nodes (with labels) for this type
    if dtype in ("diagrams", "graphviz"):
        all_nodes = parse_dot_nodes(src_path)
    elif dtype == "mermaid":
        all_nodes = parse_mmd_nodes(src_path)
    elif dtype == "plantuml":
        all_nodes = parse_puml_nodes(src_path)
    else:
        all_nodes = {}

    for nid, lbl in all_nodes.items():
        if nid in containment and lbl:
            label_to_location[lbl] = containment[nid]

    def _get_location(node_name: str) -> str:
        return label_to_location.get(node_name, "None")

    # 3. Parse arrows from SVG + source
    try:
        if dtype == "diagrams":
            arrows, vb = parse_diagrams_svg_arrows(svg_path, src_path)
        elif dtype == "graphviz":
            arrows, vb = parse_dot_svg_arrows(svg_path, src_path)
        elif dtype == "mermaid":
            arrows, vb = parse_mermaid_svg_arrows(svg_path, src_path)
        elif dtype == "plantuml":
            arrows, vb = parse_puml_svg_arrows(svg_path, src_path)
        else:
            return None
    except Exception as e:
        print(f"    SVG parse error {sample_id}: {e}")
        return None

    if not arrows:
        return None

    vb_x, vb_y, vb_w, vb_h = vb

    # 4. Build virtual node resolver
    all_edges = [(a["src_id"], a["tgt_id"], a["label"]) for a in arrows]
    vnode_resolver = build_vnode_resolver(all_nodes, all_edges)

    # 5. Get PNG image from pre-rendered PNGs
    png_path_pre = png_dir / f"{src_path.stem}.png"
    try:
        if png_path_pre.exists():
            png_img = Image.open(png_path_pre).convert("RGB")
            img_w, img_h = png_img.size
            sx = img_w / vb_w if vb_w > 0 else 1.0
            sy = img_h / vb_h if vb_h > 0 else 1.0
        elif dtype == "mermaid":
            mermaid_png = png_dir / f"{src_path.stem}.png"
            png_img, sx, sy = mermaid_to_png(src_path, mermaid_png, vb)
        else:
            png_img, scale = svg_to_png_cairosvg(svg_path, vb)
            sx = sy = scale
    except Exception as e:
        print(f"    PNG error {sample_id}: {e}")
        return None

    img_w, img_h = png_img.size

    # 6. Annotate each arrow (resolve virtual nodes)
    out_dir = OUTPUT_ROOT / sample_id
    ann_dir = out_dir / "annotated_images"
    ann_dir.mkdir(parents=True, exist_ok=True)

    proc_arrows = []
    arrow_counter = 0

    for arr in arrows:
        src_id, tgt_id = arr["src_id"], arr["tgt_id"]
        src_name, tgt_name = arr["src_name"], arr["tgt_name"]
        condition = arr["label"] if arr["label"] else "None"

        # Resolve virtual nodes
        # Case 1: both real → normal
        # Case 2: src real, tgt virtual → target = each real successor of tgt
        # Case 3: src virtual, tgt real → source = each real predecessor of src
        # Case 4: both virtual → skip
        if src_name and tgt_name:
            resolved_pairs = [(src_name, tgt_name, condition)]
        elif src_name and not tgt_name and tgt_id in vnode_resolver:
            _, successors = vnode_resolver[tgt_id]
            resolved_pairs = [
                (src_name, all_nodes.get(s, ""),
                 (cond or condition) if cond else condition)
                for s, cond in successors if all_nodes.get(s, "")
            ]
        elif not src_name and tgt_name and src_id in vnode_resolver:
            predecessors, _ = vnode_resolver[src_id]
            resolved_pairs = [
                (all_nodes.get(p, ""), tgt_name,
                 (cond or condition) if cond else condition)
                for p, cond in predecessors if all_nodes.get(p, "")
            ]
        else:
            continue  # both virtual or unresolvable

        if not resolved_pairs:
            continue

        # Compute pixel coordinates (same for all resolved pairs)
        px_tip_x  = (arr["tip_vx"]  - vb_x) * sx
        px_tip_y  = (arr["tip_vy"]  - vb_y) * sy
        px_prev_x = (arr["prev_vx"] - vb_x) * sx
        px_prev_y = (arr["prev_vy"] - vb_y) * sy
        px_tip_x  = max(0, min(img_w - 1, px_tip_x))
        px_tip_y  = max(0, min(img_h - 1, px_tip_y))

        bbox = _arrowhead_bbox(px_tip_x, px_tip_y, px_prev_x, px_prev_y)
        bbox_padded = [
            max(0, round(bbox[0] - BOX_PADDING)),
            max(0, round(bbox[1] - BOX_PADDING)),
            min(img_w - 1, round(bbox[2] + BOX_PADDING)),
            min(img_h - 1, round(bbox[3] + BOX_PADDING)),
        ]

        ann_img = annotate_arrowhead(png_img,
                                     px_tip_x, px_tip_y, px_prev_x, px_prev_y)

        # Combine all resolved pairs into one record per arrow
        arrow_counter += 1
        out_path = ann_dir / f"{sample_id}_arrow_{arrow_counter}.png"
        ann_img.save(str(out_path), "PNG")

        if len(resolved_pairs) > 1:
            src_nodes = " || ".join(p[0] for p in resolved_pairs)
            tgt_nodes = " || ".join(p[1] for p in resolved_pairs)
            conditions = " || ".join(p[2] for p in resolved_pairs)
            src_locs = " || ".join(_get_location(p[0]) for p in resolved_pairs)
            tgt_locs = " || ".join(_get_location(p[1]) for p in resolved_pairs)
        else:
            src_nodes = resolved_pairs[0][0]
            tgt_nodes = resolved_pairs[0][1]
            conditions = resolved_pairs[0][2]
            src_locs = _get_location(resolved_pairs[0][0])
            tgt_locs = _get_location(resolved_pairs[0][1])

        proc_arrows.append({
            "arrow_id": arrow_counter,
            "source_node": src_nodes,
            "source_location": src_locs,
            "condition": conditions,
            "target_node": tgt_nodes,
            "target_location": tgt_locs,
            "arrowhead_bbox": bbox_padded,
            "annotated_image_path": str(out_path.absolute()),
        })

    # Per-sample JSON
    if proc_arrows:
        with open(out_dir / f"{sample_id}.json", "w", encoding="utf-8") as f:
            json.dump({"sample_id": sample_id, "arrows": proc_arrows},
                      f, indent=2, ensure_ascii=False)

    return proc_arrows


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def _set_output_root(path: Path):
    global OUTPUT_ROOT
    OUTPUT_ROOT = path


TRAINING_DATA_ROOT = Path("<DATA_ROOT>/data_4_training")


def main():
    parser = argparse.ArgumentParser(description="Generate FlowGen arrow-triplet data")
    parser.add_argument("--types", nargs="+",
                        default=["diagrams", "graphviz", "mermaid", "plantuml"])
    parser.add_argument("--difficulties", nargs="+",
                        default=["easy", "medium", "hard"])
    args = parser.parse_args()

    types = [t for t in args.types if t in SRC_EXT]

    # One output dir per difficulty; all samples flat inside (no train/val split).
    # Downstream tooling (format_data.py for SFT, gen_coco_for_sam3.py for detection)
    # does its own splitting/sampling.
    for diff in args.difficulties:
        print(f"\n{'=' * 60}")
        print(f"  Difficulty: {diff}  (types {types})")
        print(f"{'=' * 60}")

        output = TRAINING_DATA_ROOT / f"flowgen_{diff}"
        output.mkdir(parents=True, exist_ok=True)
        _set_output_root(output)

        for dtype in types:
            print(f"\n  ── {dtype} / {diff} ──")

            src_dir = FLOWGEN_ROOT / "train" / dtype / diff
            ext = SRC_EXT[dtype]
            src_files = sorted(src_dir.glob(f"*{ext}"))
            print(f"  {len(src_files)} source files")

            svg_dir = FLOWGEN_ROOT / f"train_{dtype}_{diff}_svg"
            png_dir = FLOWGEN_ROOT / f"train_{dtype}_{diff}_png"

            processed = skipped = total_arrows = 0
            for src_path in src_files:
                sample_id = f"{dtype}_{diff}_{src_path.stem}"
                arrows = process_sample(src_path, svg_dir, png_dir, dtype, sample_id)

                if not arrows:
                    skipped += 1
                    if skipped <= 5:
                        print(f"    skip {src_path.stem}")
                    continue

                processed += 1
                total_arrows += len(arrows)
                if processed % 50 == 0:
                    print(f"    processed {processed}  skipped {skipped}  arrows {total_arrows}")

            print(f"  => {dtype}: processed {processed}, skipped {skipped}, arrows {total_arrows}")

        print(f"\n  [{diff}] output → {output}")

    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
