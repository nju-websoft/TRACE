# Convert raw intermediate data into trainable formats (works for both arrow and triplet).

import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Default triplet prompt
DEFAULT_TRIPLET_PROMPT = """<image>Please describe all the information in the flowchart image in the form of triplets: For each arrow labeled with X directed from node A to node B, output a triplet: <A, X, B>; For each unlabeled arrow directed from node A to node B, output a triplet: <A, connectedTo, B>; For each node A fully inside node group B, output a triplet <A, partOf, B>.

The following triplets are example outputs:
<Food Packaging Improvement, secures, Quantum Computing Integration>
<Graphene Production, catalyzes, Nanosensors>
<Graphene Production, partOf, Drug Delivery Systems>
<Nanocomposites, connectedTo, Spectroscopy>
<Nanodevice Fabrication, connectedTo, Graphene Production>
<Nanosensors, connectedTo, Nanocomposites>
<Spectroscopy, educates, Nanodevice Fabrication>"""

# Import the arrow prompts from lora/prompt.py (ensure repo root is on sys.path).
# Per-dataset prompts: bpmn / flowgen carry pool-lane / group "location" fields.
try:
    import sys, os
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from lora.prompt import get_triplet, get_bpmn_triplet, get_flowgen_triplet
    _PROMPTS_OK = True
except ImportError:
    _PROMPTS_OK = False
    _FALLBACK_PROMPT = "<image>Please analyze this flowchart and extract arrow information."
    print("⚠️  lora/prompt.py not found, using a default arrow extraction prompt")


def format_list_text(data_list):
    """
    Normalize a list field:
    1. empty or None -> ['None']
    2. a list -> stringify each item and strip whitespace
    """
    if not data_list:
        return ["None"]
    
    if isinstance(data_list, list):
        processed = [str(t).strip() for t in data_list if t]
        if not processed:
            return ["None"]
        return processed
    
    return [str(data_list).strip()]


def _dataset_kind(name: str) -> str:
    """Classify a dataset name into a prompt family."""
    n = (name or "").lower()
    if "bpmn" in n:
        return "bpmn"
    if "flowgen" in n:
        return "flowgen"
    return "generic"


def arrow_prompt_for(name: str) -> str:
    """Pick the per-dataset arrow (TRACE) prompt."""
    if not _PROMPTS_OK:
        return _FALLBACK_PROMPT
    kind = _dataset_kind(name)
    if kind == "bpmn":
        return get_bpmn_triplet()
    if kind == "flowgen":
        return get_flowgen_triplet()
    return get_triplet()


def build_arrow_answer(name: str, arrow: dict) -> str:
    """Build the assistant target text for one arrow.
    bpmn / flowgen emit 5 lines (with source_location/target_location); others 3 lines.
    """
    src  = " || ".join(format_list_text(arrow.get('source_node', arrow.get('start_point', []))))
    tgt  = " || ".join(format_list_text(arrow.get('target_node', arrow.get('end_point', []))))
    cond = " || ".join(format_list_text(arrow.get('condition', [])))
    if _dataset_kind(name) in ("bpmn", "flowgen"):
        src_loc = " || ".join(format_list_text(arrow.get('source_location', [])))
        tgt_loc = " || ".join(format_list_text(arrow.get('target_location', [])))
        return (f"source_node: {src}\n"
                f"source_location: {src_loc}\n"
                f"condition: {cond}\n"
                f"target_node: {tgt}\n"
                f"target_location: {tgt_loc}")
    return (f"source_node: {src}\n"
            f"target_node: {tgt}\n"
            f"condition: {cond}")


def _as_text(v) -> str:
    """Normalize a node/condition/location field (str or list) to a ' || '-joined string."""
    if isinstance(v, list):
        return " || ".join(str(x).strip() for x in v if x is not None and str(x).strip())
    return str(v).strip() if v is not None else ""


def build_image_triplets(arrows: list) -> list:
    """Whole-image E2E triplets from an arrow list.
    Emits connection triplets (<A, relation, B>) and, when source_location /
    target_location are present (bpmn/flowgen), containment triplets (<A, partOf, loc>).
    Handles both string and list fields and ' || '-separated multi-arrows; deduplicates.
    """
    triplets, seen = [], set()
    for arr in arrows:
        src = _as_text(arr.get("source_node", arr.get("start_point", "")))
        tgt = _as_text(arr.get("target_node", arr.get("end_point", "")))
        if not src or not tgt:
            continue
        cond    = _as_text(arr.get("condition", "")) or "None"
        src_loc = _as_text(arr.get("source_location", "")) or "None"
        tgt_loc = _as_text(arr.get("target_location", "")) or "None"

        srcs     = [s.strip() for s in src.split(" || ")]
        tgts     = [t.strip() for t in tgt.split(" || ")]
        conds    = [c.strip() for c in cond.split(" || ")]
        src_locs = [l.strip() for l in src_loc.split(" || ")]
        tgt_locs = [l.strip() for l in tgt_loc.split(" || ")]

        n = max(len(srcs), len(tgts), len(conds))
        if len(srcs) == 1:     srcs *= n
        if len(tgts) == 1:     tgts *= n
        if len(conds) == 1:    conds *= n
        if len(src_locs) == 1: src_locs *= n
        if len(tgt_locs) == 1: tgt_locs *= n

        def _add(trip):
            if trip not in seen:
                seen.add(trip)
                triplets.append(trip)

        for i in range(min(len(srcs), len(tgts))):
            s, t = srcs[i], tgts[i]
            if not s or not t:
                continue
            c  = conds[i]    if i < len(conds)    else "None"
            sl = src_locs[i] if i < len(src_locs) else "None"
            tl = tgt_locs[i] if i < len(tgt_locs) else "None"
            rel = c if (c and c.lower() != "none") else "connectedTo"
            _add(f"<{s}, {rel}, {t}>")
            if sl and sl.lower() != "none":
                _add(f"<{s}, partOf, {sl}>")
            if tl and tl.lower() != "none":
                _add(f"<{t}, partOf, {tl}>")
    return triplets


def convert_to_arrow_format(sampled_data, output_file, prefix):
    """
    Format 1: arrow-level (TRACE).
    One training sample per arrow. Output format:
    source_node: xxx
    target_node: xxx
    condition: xxx
    `sampled_data` is the in-memory manifest (list of file-items).
    """
    print("\n" + "="*60)
    print("Converting to arrow-level (TRACE) training format...")
    print("="*60)

    finetune_data = []
    json_cache = {}
    processed_count = 0
    skipped_count = 0

    for file_item in tqdm(sampled_data, desc=f"  {prefix}"):
        try:
            json_path = file_item['json_path']
            folder_name = file_item['folder_name']

            if json_path not in json_cache:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_cache[json_path] = json.load(f)
            arrows = json_cache[json_path].get('arrows', [])
            if not arrows:
                skipped_count += 1
                continue

            for arrow in arrows:
                arrow_id = arrow.get('arrow_id', 'unknown')
                image_path = arrow.get('annotated_image_path', file_item['image_path'])
                image_path = resolve_image_path(image_path, Path(json_path))
                if not image_path:
                    skipped_count += 1
                    continue
                data_entry = {
                    "id": f"{folder_name}_{arrow_id}",
                    "image": image_path,
                    "conversations": [
                        {"from": "user", "value": arrow_prompt_for(prefix)},
                        {"from": "assistant", "value": build_arrow_answer(prefix, arrow)},
                    ],
                }
                finetune_data.append(data_entry)
                processed_count += 1
        except Exception as e:
            print(f"\u274c error processing {file_item.get('folder_name', 'unknown')}: {e}")
            skipped_count += 1
            continue

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(finetune_data, f, indent=2, ensure_ascii=False)
    print(f"\n\u2705 arrow-level training data saved to: {output_file}")
    print(f"   total arrows: {processed_count}")
    print(f"   skipped: {skipped_count}")
    print("="*60)


def convert_to_triplet_format(sampled_data, output_file, prefix):
    """
    Format 2: triplet (E2E).
    One training sample per image; output all arrows as triplets.
    `sampled_data` is the in-memory manifest (list of file-items).
    """
    print("\n" + "="*60)
    print("Converting to triplet (E2E) training format...")
    print("="*60)

    all_conversations = []
    processed_count = 0
    skipped_count = 0

    for file_item in tqdm(sampled_data, desc=f"  {prefix}"):
        try:
            json_path = file_item['json_path']
            image_path = file_item['image_path']
            folder_name = file_item['folder_name']
            if not image_path or not Path(image_path).is_file():
                skipped_count += 1
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            answer = "\n".join(build_image_triplets(json_data.get("arrows", [])))
            if not answer:
                skipped_count += 1
                continue

            all_conversations.append({
                "id": folder_name,
                "image": image_path,
                "conversations": [
                    {"from": "user", "value": DEFAULT_TRIPLET_PROMPT},
                    {"from": "assistant", "value": answer},
                ],
            })
            processed_count += 1
        except Exception as e:
            print(f"\u274c error processing {file_item.get('folder_name', 'unknown')}: {e}")
            skipped_count += 1
            continue

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    print(f"\n\u2705 triplet training data saved to: {output_file}")
    print(f"   total samples: {processed_count}")
    print(f"   skipped: {skipped_count}")
    print("="*60)


def build_manifest(data_dir, image_dir=None):
    """Scan an intermediate-data directory (one folder per sample, each holding a
    <name>.json with an 'arrows' list) and return the file-item list the converters
    consume: [{folder_name, json_path, image_path}, ...].

    The whole-image path (used by the triplet/E2E format) is resolved from the JSON's
    own 'image_path' field, else <image_dir>/<folder>.(png|jpeg), else a same-named
    image inside the sample folder. The arrow/TRACE format mainly uses each arrow's
    own 'annotated_image_path', so a missing whole image is non-fatal there.
    """
    data_dir = Path(data_dir)
    image_dir = Path(image_dir) if image_dir else None
    items = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        json_files = list(folder.glob("*.json"))
        if not json_files:
            continue
        merged = [f for f in json_files if "_merged" in f.name]
        json_path = merged[0] if merged else json_files[0]
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not data.get("arrows"):
            continue
        cands = []
        extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif')
        if image_dir:
            cands += [image_dir / f'{folder.name}{ext}' for ext in extensions]
        if data.get('image_path'):
            cands.append(Path(data['image_path']))
        cands += [folder / f'{folder.name}{ext}' for ext in extensions]
        img = next((str(c.resolve()) for c in cands if c.is_file()), '')
        items.append({
            "folder_name": folder.name,
            "json_path": str(json_path),
            "image_path": str(img),
        })
    return items


def resolve_image_path(value, json_path):
    """Resolve released relative annotation paths without rewriting dataset JSON."""
    if not value:
        return ''
    path = Path(value)
    candidates = [path, json_path.parent / path]
    # Released annotations use paths relative to data_4_training, e.g. fcb/.../arrow.jpg.
    for parent in json_path.parents:
        if parent.name == 'data_4_training':
            candidates.append(parent / path)
            if 'data_4_training/' in str(value):
                candidates.append(parent / str(value).split('data_4_training/', 1)[1])
            break
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return ''


def main():
    parser = argparse.ArgumentParser(description='Convert intermediate arrow data into training formats')
    parser.add_argument('--data-dir', required=True,
                        help='Intermediate-data directory (one folder per sample, each with a <name>.json)')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name; selects the arrow prompt (bpmn / flowgen / else generic)')
    parser.add_argument('--image-dir', default=None,
                        help='Optional whole-image base dir (used by the triplet/E2E format)')
    parser.add_argument('--format', choices=['arrow', 'triplet', 'all'], default='all',
                        help='Output format: arrow (TRACE), triplet (E2E), or all')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to write the training json(s) into')
    parser.add_argument('--output-prefix', default=None,
                        help='Output filename prefix; defaults to the dataset name')
    parser.add_argument('--triplet-output', type=Path, default=None,
                        help='Optional explicit E2E output filename (e.g. bpmn_triplet_dev.json)')
    args = parser.parse_args()

    manifest = build_manifest(args.data_dir, args.image_dir)
    if not manifest:
        print(f"❌ no samples with arrows found under {args.data_dir}")
        return

    out_dir = Path(args.output_dir)
    output_prefix = args.output_prefix or args.dataset
    print("="*60)
    print("Data format converter")
    print(f"  data dir : {args.data_dir}")
    print(f"  dataset  : {args.dataset}")
    print(f"  samples  : {len(manifest)}")
    print(f"  format   : {args.format}")
    print("="*60)

    if args.format in ('arrow', 'all'):
        convert_to_arrow_format(manifest, out_dir / f"{output_prefix}_arrow_training_all.json", args.dataset)
    if args.format in ('triplet', 'all'):
        convert_to_triplet_format(manifest, args.triplet_output or out_dir / f"{output_prefix}_triplet_training_all.json", args.dataset)
    
    print("\n" + "="*60)
    print("✅ all conversions done!")
    print("="*60)


if __name__ == "__main__":
    main()
