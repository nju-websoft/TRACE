"""Score FlowLearn predictions for the 4 short-answer sub-tasks.

Prompt format follows FlowGen Appendix E Yes/No template; ground-truth labels
are 'yes' / 'no' (and integers for the counting tasks).

Sub-tasks (ground-truth answer in parentheses):
  - Flowchart-isTrue-AtoB         (yes)
  - Flowchart-isFalse-AtoB        (no)
  - Flowchart-isTrue-betweenAB    (yes)
  - Flowchart-isFalse-betweenAB   (no)
  - Num_Nodes                     (integer)
  - Num_Arrows                    (integer)

Single metric: accuracy (per sub-task, per main task, overall).
"""

import os
import re
import json
import argparse
from collections import defaultdict


# Accept lenient surface forms — model may say "Yes", "no.", "true", etc.
_YES_TOKENS = {"yes", "true", "correct", "right"}
_NO_TOKENS = {"no", "false", "incorrect", "wrong"}
_INT_PAT = re.compile(r"-?\d+")


def parse_yesno(text: str):
    if not text:
        return None
    s = text.strip().lower()
    head = re.split(r"[^a-z]+", s, maxsplit=1)[0]
    if head in _YES_TOKENS:
        return "yes"
    if head in _NO_TOKENS:
        return "no"
    head_chunk = s[:60]
    has_y = any(re.search(rf"\b{tok}\b", head_chunk) for tok in _YES_TOKENS)
    has_n = any(re.search(rf"\b{tok}\b", head_chunk) for tok in _NO_TOKENS)
    if has_y and not has_n:
        return "yes"
    if has_n and not has_y:
        return "no"
    return None


def parse_int(text: str):
    if not text:
        return None
    m = _INT_PAT.search(text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def evaluate(pred_file: str, verbose: bool = True):
    with open(pred_file) as f:
        data = json.load(f)

    sub_stats = defaultdict(lambda: {"correct": 0, "total": 0, "unparsed": 0})

    for img_key, sub_dict in data.items():
        for sub_key, item in sub_dict.items():
            pred = item.get("predicted", "")
            gt = item.get("gt", "")
            stat = sub_stats[sub_key]
            stat["total"] += 1

            if sub_key.startswith("Flowchart-is"):
                p = parse_yesno(pred)
                if p is None:
                    stat["unparsed"] += 1
                    continue
                if p == gt.strip().lower():
                    stat["correct"] += 1
            elif sub_key in ("Num_Nodes", "Num_Arrows"):
                p = parse_int(pred)
                if p is None:
                    stat["unparsed"] += 1
                    continue
                try:
                    g = int(gt)
                except (ValueError, TypeError):
                    continue
                if p == g:
                    stat["correct"] += 1
            else:
                # Unknown sub-task — skip
                stat["total"] -= 1

    # Group by main task (strip the suffix after the last '-')
    def main_task(sub_key: str) -> str:
        if sub_key.startswith("Flowchart-isTrue") or sub_key.startswith("Flowchart-isFalse"):
            # Group by AtoB / betweenAB
            return "Arrow_" + sub_key.split("-")[-1]
        return sub_key

    grouped = defaultdict(lambda: {"correct": 0, "total": 0, "unparsed": 0})
    for sub_key, st in sub_stats.items():
        m = main_task(sub_key)
        grouped[m]["correct"] += st["correct"]
        grouped[m]["total"] += st["total"]
        grouped[m]["unparsed"] += st["unparsed"]

    overall = {"correct": 0, "total": 0, "unparsed": 0}
    for st in sub_stats.values():
        for k in overall:
            overall[k] += st[k]

    def acc(d):
        return d["correct"] / d["total"] if d["total"] else 0.0

    summary = {
        "overall": {
            "n": overall["total"],
            "accuracy": acc(overall),
            "unparsed": overall["unparsed"],
        },
        "per_task": {
            t: {"n": d["total"], "accuracy": acc(d), "unparsed": d["unparsed"]}
            for t, d in sorted(grouped.items())
        },
        "per_sub_task": {
            t: {"n": d["total"], "accuracy": acc(d), "unparsed": d["unparsed"]}
            for t, d in sorted(sub_stats.items())
        },
    }

    if verbose:
        print("=" * 64)
        print(f"FlowLearn evaluation  ({pred_file})")
        print("=" * 64)
        print(f"Overall   N={overall['total']:>5d}  accuracy={summary['overall']['accuracy']:.4f}  "
              f"unparsed={overall['unparsed']}")
        print("-" * 64)
        print(f"  {'task':<26s} {'N':>5s}  {'acc':>6s}  unparsed")
        for t, m in summary["per_task"].items():
            print(f"  {t:<26s} {m['n']:>5d}  {m['accuracy']:>6.4f}  {m['unparsed']}")
        print("-" * 64)
        for t, m in summary["per_sub_task"].items():
            print(f"    {t:<28s} {m['n']:>5d}  {m['accuracy']:>6.4f}  {m['unparsed']}")
        print("=" * 64)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-file", required=True)
    ap.add_argument("--output-file", default=None)
    args = ap.parse_args()

    summary = evaluate(args.prediction_file)
    out = args.output_file or os.path.join(
        os.path.dirname(args.prediction_file),
        os.path.basename(args.prediction_file).rsplit(".", 1)[0] + "_score.json",
    )
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
