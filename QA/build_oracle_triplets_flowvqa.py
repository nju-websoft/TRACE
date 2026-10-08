"""Build oracle triplets from FlowVQA mermaid code.

For each entry in test_full.json, parse the `mermaid` field and emit
`<oracle_dir>/<key>/arrow_triplets.json` in the same format as the LoRA
extraction output, so QA inference scripts can pick them up via --triplets-dir.

Note: this is the upper-bound oracle. We keep the original node text
(not normalized) so the model gets readable context.
"""

import os
import re
import json
import argparse


# Match a node definition: <id><opening><raw_text><closing>
# Various node shapes Mermaid supports: [], (), {}, [/.../], [(...)], etc.
_NODE_RE = re.compile(
    r"([A-Za-z0-9_]+)\s*"
    r"(?:\[\[|\(\(|\[\(|\(\[|\[\\|\[/|\[|\(|\{)"
    r"\s*[\"']?(.*?)[\"']?\s*"
    r"(?:\]\]|\)\)|\)\]|\]\)|\\\]|/\]|\}|\]|\))"
)

# Match an edge: <src> [--> | ==> | -.-> | -- ... --> ...] <dst>
# We also capture an optional |"label"| or |label|
_EDGE_RE = re.compile(
    r"\b([A-Za-z0-9_]+)\b"
    r"\s*(?:==>|-->|-\.->|--[^>|\n]*-->)"
    r"\s*(?:\|\s*[\"']?(.*?)[\"']?\s*\|)?"
    r"\s*\b([A-Za-z0-9_]+)\b"
)


def parse_mermaid(mermaid: str):
    """Return list of {'source','end','condition'} dicts in original text."""
    if not mermaid:
        return []
    code = mermaid.replace('\\"', '"').replace("\\'", "'")

    # Build node_id -> display_text. If a node id appears multiple times keep
    # the longest text.
    node_map = {}
    for line in code.splitlines():
        for nid, raw in _NODE_RE.findall(line):
            text = (raw or "").strip()
            if not text:
                continue
            if nid not in node_map or len(text) > len(node_map[nid]):
                node_map[nid] = text

    triplets = []
    seen = set()  # (src_text, end_text, cond) dedupe
    for line in code.splitlines():
        for src_id, label, dst_id in _EDGE_RE.findall(line):
            src = node_map.get(src_id, src_id)
            dst = node_map.get(dst_id, dst_id)
            cond = (label or "").strip() or None
            key = (src, dst, cond)
            if key in seen:
                continue
            seen.add(key)
            triplets.append({
                "arrow_id": len(triplets) + 1,
                "arrow_label": f"H{len(triplets) + 1}",
                "source": src,
                "end": dst,
                "condition": cond,
            })
    return triplets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-json",
                    default="<DATA_ROOT>/flowvqa/Data/test_full.json")
    ap.add_argument("--output-dir", required=True,
                    help="Will be created. One <key>/arrow_triplets.json per entry.")
    args = ap.parse_args()

    with open(args.test_json) as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    n = 0
    n_empty = 0
    edge_total = 0
    for k, sample in data.items():
        mermaid = sample.get("mermaid", "")
        triplets = parse_mermaid(mermaid)
        if not triplets:
            n_empty += 1
        edge_total += len(triplets)
        out_dir = os.path.join(args.output_dir, k)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "arrow_triplets.json"), "w") as f:
            json.dump(triplets, f, indent=2, ensure_ascii=False)
        n += 1

    print(f"wrote {n} entries to {args.output_dir}  "
          f"(empty={n_empty}, total_edges={edge_total}, avg={edge_total/max(1,n):.1f}/img)")


if __name__ == "__main__":
    main()
