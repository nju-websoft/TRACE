"""eval_TRACE.py — evaluator for the TRACE pipeline (v1 outputs).

Reads predictions from a directory of subdirs (one per image, each containing
``arrow_triplets.json``) and computes F1 against per-dataset ground truth.

    python -m lora.eval_TRACE --dataset <name> --output-dir <pred_dir> [extra args]

Datasets: bpmn, cbd, fca, fcb, flowgen, flowlearn, flowvqa
"""

import argparse
import glob
import json
import os
import re
import sys

import Levenshtein

try:
    from .scope import select_ground_truth
except ImportError:
    from scope import select_ground_truth


# ─── Shared helpers ─────────────────────────────────────────────────────────

def edit_similarity(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    return 1 - Levenshtein.distance(s1, s2) / max_len


def _relaxed_match(pred, gt, es_thr=0.85):
    """Return (exact_hits, relaxed_hits, relaxed_matched_pairs)."""
    correct_set = pred & gt
    exact_hits = len(correct_set)
    rem_gt = sorted(gt - correct_set)
    rem_pred = sorted(pred - correct_set)
    relaxed_hits = exact_hits
    relaxed_matched = []  # list of (gt_triplet, pred_triplet)
    matched_pred_idx = set()
    for g in rem_gt:
        g_h, g_r, g_t = g
        for i, p in enumerate(rem_pred):
            if i in matched_pred_idx:
                continue
            p_h, p_r, p_t = p
            if (edit_similarity(p_h, g_h) >= es_thr and
                edit_similarity(p_r, g_r) >= es_thr and
                edit_similarity(p_t, g_t) >= es_thr):
                relaxed_hits += 1
                matched_pred_idx.add(i)
                relaxed_matched.append((g, p))
                break
    return exact_hits, relaxed_hits, relaxed_matched


def _prf(hit, p_total, g_total):
    p = hit / p_total if p_total else 0
    r = hit / g_total if g_total else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return p, r, f1


def _load_predictions(pred_dir, normalize_text, normalize_relation, with_location=False):
    """Read arrow_triplets.json files. Accepts both new (source_node/target_node)
    and legacy (source/end) field names. Returns dict[file_key] -> set of triplets.
    """
    preds = {}
    pattern = os.path.join(pred_dir, "*", "arrow_triplets.json")
    for fpath in glob.glob(pattern):
        file_key = os.path.basename(os.path.dirname(fpath))
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        triplets = set()
        for item in data:
            src_raw = item.get("source_node", item.get("source", ""))
            tgt_raw = item.get("target_node", item.get("end", ""))
            src = normalize_text(src_raw)
            tgt = normalize_text(tgt_raw)
            cond = normalize_relation(item.get("condition", ""))
            if src and tgt and tgt != "pending":
                triplets.add((src, cond, tgt))
            if with_location:
                src_loc = normalize_text(item.get("source_location", ""))
                tgt_loc = normalize_text(item.get("target_location", ""))
                if src and src_loc and src_loc != "none":
                    triplets.add((src, "partof", src_loc))
                if tgt and tgt_loc and tgt_loc != "none":
                    triplets.add((tgt, "partof", tgt_loc))
        preds[file_key] = triplets
    return preds


def _evaluate_simple(dataset_name, pred_data, gt_data):
    """Connection-only evaluation. Returns results dict."""
    common = sorted(gt_data)
    total_correct = total_relaxed = total_pred = total_gt = 0
    for fname in common:
        gt = gt_data[fname]
        pred = pred_data.get(fname, set())
        total_gt += len(gt)
        total_pred += len(pred)
        e, r, _ = _relaxed_match(pred, gt)
        total_correct += e
        total_relaxed += r

    p, r, f1 = _prf(total_correct, total_pred, total_gt)
    rp, rr, rf1 = _prf(total_relaxed, total_pred, total_gt)
    return {
        "dataset": dataset_name,
        "processed_files": len(common),
        "missing_prediction_files": len(set(gt_data) - set(pred_data)),
        "gt_files": len(gt_data),
        "pred_files": len(pred_data),
        "total_gt": total_gt,
        "total_predicted": total_pred,
        "exact_hits": total_correct,
        "relaxed_hits": total_relaxed,
        "standard_metrics": {"precision": p, "recall": r, "f1": f1},
        "relaxed_metrics": {"precision": rp, "recall": rr, "f1": rf1},
        "f1_score": (rf1 + f1) / 2,
        "score": (rf1 + f1) / 2,
        "relaxed_f1": rf1,
    }


def _evaluate_with_containment(dataset_name, pred_data, gt_data):
    """Split triplets into connection (rel != 'partof') and containment, score each."""
    common = sorted(gt_data)
    total_correct = total_relaxed = total_pred = total_gt = 0
    conn_correct = conn_relaxed = conn_pred = conn_gt = 0
    cont_correct = cont_relaxed = cont_pred = cont_gt = 0

    for fname in common:
        gt = gt_data[fname]
        pred = pred_data.get(fname, set())
        total_gt += len(gt)
        total_pred += len(pred)

        gt_conn = {t for t in gt if t[1] != "partof"}
        gt_cont = {t for t in gt if t[1] == "partof"}
        pred_conn = {t for t in pred if t[1] != "partof"}
        pred_cont = {t for t in pred if t[1] == "partof"}
        conn_gt += len(gt_conn)
        conn_pred += len(pred_conn)
        cont_gt += len(gt_cont)
        cont_pred += len(pred_cont)

        correct_set = pred & gt
        total_correct += len(correct_set)
        conn_correct += len(correct_set & gt_conn)
        cont_correct += len(correct_set & gt_cont)

        _, file_relaxed, relaxed_pairs = _relaxed_match(pred, gt)
        total_relaxed += file_relaxed
        for (g_h, g_r, g_t), _ in relaxed_pairs:
            if g_r == "partof":
                cont_relaxed += 1
            else:
                conn_relaxed += 1
        # carry over exact-match contributions into per-bucket relaxed counts
        conn_relaxed += len(correct_set & gt_conn)
        cont_relaxed += len(correct_set & gt_cont)

    p, r, f1 = _prf(total_correct, total_pred, total_gt)
    rp, rr, rf1 = _prf(total_relaxed, total_pred, total_gt)
    cp, cr, cf1 = _prf(conn_correct, conn_pred, conn_gt)
    crp, crr, crf1 = _prf(conn_relaxed, conn_pred, conn_gt)
    tp, tr, tf1 = _prf(cont_correct, cont_pred, cont_gt)
    trp, trr, trf1 = _prf(cont_relaxed, cont_pred, cont_gt)

    return {
        "dataset": dataset_name,
        "processed_files": len(common),
        "missing_prediction_files": len(set(gt_data) - set(pred_data)),
        "gt_files": len(gt_data),
        "pred_files": len(pred_data),
        "total_gt": total_gt,
        "total_predicted": total_pred,
        "exact_hits": total_correct,
        "relaxed_hits": total_relaxed,
        "overall": {
            "standard": {"precision": p, "recall": r, "f1": f1},
            "relaxed": {"precision": rp, "recall": rr, "f1": rf1},
        },
        "connection": {
            "gt": conn_gt, "pred": conn_pred,
            "standard": {"precision": cp, "recall": cr, "f1": cf1},
            "relaxed": {"precision": crp, "recall": crr, "f1": crf1},
        },
        "containment": {
            "gt": cont_gt, "pred": cont_pred,
            "standard": {"precision": tp, "recall": tr, "f1": tf1},
            "relaxed": {"precision": trp, "recall": trr, "f1": trf1},
        },
        "f1_score": (rf1 + f1) / 2,
        "score": (rf1 + f1) / 2,
        "relaxed_f1": rf1,
    }


def _write_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _print_score(results):
    rf1 = results.get("relaxed_f1", results.get("f1_score", 0.0))
    print(f"F1 Score: {results.get('f1_score', 0.0):.4f}")
    print(f"Relaxed F1: {rf1:.4f}")


# ─── Per-dataset normalize functions ─────────────────────────────────────────

# Default (bpmn, fca, fcb, flowgen, cbd-arrow style): lowercase + strip
def _lower_strip(text):
    if not text:
        return ""
    return str(text).strip().lower()


# Aggressive alphanum-only (cbd v1 / flowlearn / flowvqa)
def _alnum_only(text):
    if not text:
        return ""
    text = str(text).replace("**", "").replace("`", "").replace("_", " ").lower()
    return re.sub(r"[^a-z0-9]", "", text).strip()


# fca/fcb: handle empty marker "o"/"0" → "no text"
def _fca_node_text(text):
    cleaned = _lower_strip(text)
    if cleaned in ("o", "0"):
        return "no text"
    return cleaned


# flowvqa: keep hyphen, strip spaces
def _flowvqa_text(text):
    if not text:
        return ""
    text = str(text).replace("_", " ").lower().replace('\\"', '"')
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    return text.replace(" ", "").strip()


# Default relation normalizer (most datasets share this)
RELATION_MAP_DEFAULT = {
    "connected_with": "connect", "connected": "connect",
    "connectedto": "connect", "": "connect", "none": "connect",
}


def _normalize_relation_default(rel):
    if rel is None:
        return "connect"
    raw = str(rel).strip().lower()
    return RELATION_MAP_DEFAULT.get(raw, raw)


RELATION_MAP_FLOWLEARN = {
    "connectedto": "connect", "connected_with": "connect",
    "": "connect", "none": "connect",
}


def _normalize_relation_flowlearn(rel):
    if rel is None:
        return "connect"
    raw = str(rel).strip().lower()
    return RELATION_MAP_FLOWLEARN.get(raw, raw)


RELATION_MAP_FLOWVQA = {
    "connected_with": "connect", "connected": "connect",
    "": "connect", "none": "connect",
    "yes": "yes", "no": "no", "true": "true", "false": "false",
}


def _normalize_relation_flowvqa(rel):
    if rel is None:
        return "connect"
    raw = str(rel).replace('"', '').replace("'", "").strip().lower()
    return RELATION_MAP_FLOWVQA.get(raw, raw)


# ─── Per-dataset GT loaders ─────────────────────────────────────────────────

def _gt_bpmn(gt_root, normalize_text, normalize_relation):
    gt = {}
    if not os.path.isdir(gt_root):
        return gt
    for subdir in os.listdir(gt_root):
        json_path = os.path.join(gt_root, subdir, f"{subdir}.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        triplets = set()
        for item in data.get("arrows", []):
            src = normalize_text(item.get("source_node", ""))
            tgt = normalize_text(item.get("target_node", ""))
            cond = normalize_relation(item.get("condition", ""))
            src_loc = normalize_text(item.get("source_location", ""))
            tgt_loc = normalize_text(item.get("target_location", ""))
            if src and tgt:
                triplets.add((src, cond, tgt))
            if src and src_loc and src_loc != "none":
                triplets.add((src, "partof", src_loc))
            if tgt and tgt_loc and tgt_loc != "none":
                triplets.add((tgt, "partof", tgt_loc))
        gt[subdir] = triplets
    return gt


def _gt_cbd(summary_file, triplet_file, normalize_text, normalize_relation):
    file_names = []
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            file_names.append(json.loads(line)["file_name"])
    with open(triplet_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    triplet_re = re.compile(r"<H>\s*(.*?)\s*<R>\s*(.*?)\s*<T>\s*(.*?)\s*(?=<H>|$)")
    gt = {}
    for fname, line in zip(file_names, lines):
        triplets = set()
        for h, r, t in triplet_re.findall(line):
            triplets.add((normalize_text(h), normalize_relation(r), normalize_text(t)))
        gt[os.path.splitext(fname)[0]] = triplets
    return gt


def _gt_fc(gt_root, normalize_text, normalize_relation):
    """FC_A / FC_B share the same per-image GT format."""
    gt = {}
    if not os.path.isdir(gt_root):
        return gt
    for subdir in os.listdir(gt_root):
        json_path = os.path.join(gt_root, subdir, f"{subdir}.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        triplets = set()
        for item in data.get("arrows", []):
            src = normalize_text(item.get("start_point", [""])[0])
            end = normalize_text(item.get("end_point", [""])[0])
            cond = item.get("condition", [""])
            cond = "" if not cond else cond[0]
            cond = normalize_relation(cond)
            if src and end:
                triplets.add((src, cond, end))
        gt[subdir] = triplets
    return gt


def _gt_flowgen(gt_root, normalize_text, normalize_relation):
    """FlowGen: per-sample dirs + image_names.json mapping stem → sample_id."""
    gt = {}
    if not os.path.isdir(gt_root):
        return gt

    img_map = {}
    img_names_path = os.path.join(gt_root, "image_names.json")
    if os.path.exists(img_names_path):
        with open(img_names_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            stem = os.path.splitext(e["image_name"])[0]
            img_map[stem] = e["sample_id"]
    # reverse lookup: sample_id → stem
    sid_to_stem = {v: k for k, v in img_map.items()}

    for subdir in os.listdir(gt_root):
        json_path = os.path.join(gt_root, subdir, f"{subdir}.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        triplets = set()
        for item in data.get("arrows", []):
            src = normalize_text(item.get("source_node", ""))
            tgt = normalize_text(item.get("target_node", ""))
            cond = normalize_relation(item.get("condition", ""))
            src_loc = normalize_text(item.get("source_location", ""))
            tgt_loc = normalize_text(item.get("target_location", ""))
            if src and tgt:
                triplets.add((src, cond, tgt))
            if src and src_loc and src_loc != "none":
                triplets.add((src, "partof", src_loc))
            if tgt and tgt_loc and tgt_loc != "none":
                triplets.add((tgt, "partof", tgt_loc))
        # Key by image stem (matches prediction file_key)
        gt_key = sid_to_stem.get(subdir, subdir)
        gt[gt_key] = triplets
    return gt


def _gt_flowlearn(gt_dir, normalize_text, normalize_relation):
    gt = {}
    if not os.path.isdir(gt_dir):
        return gt
    triplet_re = re.compile(r"<\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*>")
    for filename in os.listdir(gt_dir):
        if not filename.endswith(".txt"):
            continue
        code_id = os.path.splitext(filename)[0]
        triplets = set()
        try:
            with open(os.path.join(gt_dir, filename), "r", encoding="utf-8") as f:
                for line in f:
                    m = triplet_re.search(line)
                    if m:
                        s, r, t = m.groups()
                        s_n = normalize_text(s)
                        r_n = normalize_relation(r)
                        t_n = normalize_text(t)
                        if s_n and t_n:
                            triplets.add((s_n, r_n, t_n))
        except Exception:
            continue
        if triplets:
            gt[code_id] = triplets
    return gt


def _gt_flowvqa(gt_file, normalize_text, normalize_relation):
    if not os.path.exists(gt_file):
        return {}
    node_pat = re.compile(
        r'([A-Za-z0-9_]+)\s*'
        r'(?:\[\/|\[\\|\[\(|\[\[|\(\[|\(\(|\{|\[|\()'
        r'\s*["\']?(.*?)["\']?\s*'
        r'(?:\]\]|\/\]|\\\]|\)\]|\)\)|\}|\)|\])'
    )
    arrow_pat = re.compile(
        r'([A-Za-z0-9_]+)\s*-->\s*(?:\|["\']?(.*?)["\']?\|\s*)?([A-Za-z0-9_]+)'
    )

    def parse_mermaid(code):
        code = code.replace('\\"', '"')
        lines = code.split("\n")
        node_map = {}
        for line in lines:
            for node_id, raw in node_pat.findall(line):
                text = normalize_text(raw)
                if not text:
                    continue
                if node_id not in node_map or len(text) > len(node_map[node_id]):
                    node_map[node_id] = text
        triplets = set()
        for line in lines:
            for src_id, raw_cond, tgt_id in arrow_pat.findall(line):
                src_text = node_map.get(src_id, "")
                tgt_text = node_map.get(tgt_id, "")
                cond_norm = normalize_relation(raw_cond) if raw_cond else "connect"
                if src_text and tgt_text:
                    triplets.add((src_text, cond_norm, tgt_text))
        return triplets

    gt = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, value in iterator:
        real_key = str(key)
        if isinstance(value, dict) and "id" in value:
            real_key = str(value["id"])
        elif isinstance(raw, dict):
            real_key = key
        mermaid = value.get("mermaid", "")
        if mermaid:
            triplets = parse_mermaid(mermaid)
            if triplets:
                gt[real_key] = triplets
    return gt


# ─── CLI dispatch ───────────────────────────────────────────────────────────

DATASETS = ["bpmn", "cbd", "fca", "fcb", "flowgen", "flowlearn", "flowvqa"]

GT_DEFAULTS = {
    "bpmn":      "<DATA_ROOT>/data_4_training/bpmn_test/",
    "fca":       "<DATA_ROOT>/FC_A/",
    "fcb":       "<DATA_ROOT>/FC_B",
    "flowgen":   None,  # must be supplied
    "flowlearn": "<DATA_ROOT>/flowlearn/mermaid_word/jpeg",
    "flowvqa":   "<DATA_ROOT>/flowvqa/Data/test_full.json",
    "cbd_summary":  "<DATA_ROOT>/Block-Diagram-Datasets/Computerized_block_diagrams(CBD)/test_summary.jsonl",
    "cbd_triplets": "<DATA_ROOT>/Block-Diagram-Datasets/Computerized_block_diagrams(CBD)/test_triplets.txt",
}


def main():
    parser = argparse.ArgumentParser(
        description="TRACE pipeline evaluator (v1) — predictions from per-image arrow_triplets.json",
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--output-dir", "--prediction-dir", type=str, required=True,
                        dest="output_dir")
    parser.add_argument("--gt-root", type=str, default=None)
    parser.add_argument("--gt-summary", type=str, default=GT_DEFAULTS["cbd_summary"])
    parser.add_argument("--gt-triplets", type=str, default=GT_DEFAULTS["cbd_triplets"])
    parser.add_argument("--gt-dir", type=str, default=GT_DEFAULTS["flowlearn"])
    parser.add_argument("--gt-file", type=str, default=GT_DEFAULTS["flowvqa"])
    parser.add_argument("--image-dir", default=None, help="Original test-image directory; scopes GT independently of outputs")
    parser.add_argument("--test-json", default=None, help="Test split IDs; takes precedence over --image-dir")

    args = parser.parse_args()
    output_json = os.path.join(args.output_dir, "evaluation_results.json")
    os.makedirs(args.output_dir, exist_ok=True)

    ds = args.dataset
    try:
        if ds == "bpmn":
            gt_root = args.gt_root or GT_DEFAULTS["bpmn"]
            preds = _load_predictions(args.output_dir, _lower_strip, _normalize_relation_default, with_location=True)
            gt = _gt_bpmn(gt_root, _lower_strip, _normalize_relation_default)
            results = _evaluate_with_containment("BPMN", preds, gt)
        elif ds == "cbd":
            preds = _load_predictions(args.output_dir, _alnum_only, _normalize_relation_default)
            gt = _gt_cbd(args.gt_summary, args.gt_triplets, _alnum_only, _normalize_relation_default)
            results = _evaluate_simple("CBD", preds, gt)
        elif ds == "fca":
            gt_root = args.gt_root or GT_DEFAULTS["fca"]
            preds = _load_predictions(args.output_dir, _fca_node_text, _normalize_relation_default)
            gt = _gt_fc(gt_root, _fca_node_text, _normalize_relation_default)
            results = _evaluate_simple("FC_A", preds, gt)
        elif ds == "fcb":
            gt_root = args.gt_root or GT_DEFAULTS["fcb"]
            preds = _load_predictions(args.output_dir, _fca_node_text, _normalize_relation_default)
            gt = _gt_fc(gt_root, _fca_node_text, _normalize_relation_default)
            results = _evaluate_simple("FC_B", preds, gt)
        elif ds == "flowgen":
            if not args.gt_root:
                parser.error("--gt-root is required for flowgen")
            preds = _load_predictions(args.output_dir, _lower_strip, _normalize_relation_default, with_location=True)
            gt = _gt_flowgen(args.gt_root, _lower_strip, _normalize_relation_default)
            results = _evaluate_with_containment("FlowGen", preds, gt)
        elif ds == "flowlearn":
            preds = _load_predictions(args.output_dir, _alnum_only, _normalize_relation_flowlearn)
            gt = _gt_flowlearn(args.gt_dir, _alnum_only, _normalize_relation_flowlearn)
            results = _evaluate_simple("FlowLearn", preds, gt)
        elif ds == "flowvqa":
            preds = _load_predictions(args.output_dir, _flowvqa_text, _normalize_relation_flowvqa)
            gt = _gt_flowvqa(args.gt_file, _flowvqa_text, _normalize_relation_flowvqa)
            results = _evaluate_simple("FlowVQA", preds, gt)

        gt = select_ground_truth(gt, args.image_dir, args.test_json)
        evaluator = _evaluate_with_containment if ds in {'bpmn', 'flowgen'} else _evaluate_simple
        results = evaluator(results['dataset'], preds, gt)
        _write_results(results, output_json)
        _print_score(results)
        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
