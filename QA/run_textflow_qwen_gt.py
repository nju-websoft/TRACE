"""TextFlow-style reasoning with GROUND-TRUTH Mermaid as the intermediate
representation, and Qwen3-VL-4B as the text-only reasoner.

This is the *most generous* setting for TextFlow: it bypasses textualizer
errors entirely by giving the reasoner the source Mermaid code used to render
the image. The only variable left is the reasoner backbone.

Usage:
    python QA/run_textflow_qwen_gt.py \
        --dataset flowlearn \
        --test-json /tmp/textflow_check/data/flowlearn/test.json \
        --gt-mermaid-dir <DATA_ROOT>/flowlearn/mermaid_word/txt \
        --output-file QA/results/textflow_subset/textflow_qwen_gt_flowlearn.json \
        --base-model-path <MODEL_ROOT>/Qwen3-VL-4B-Instruct \
        --gpus 0,1,2,3
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "QA"))

from qa_runner import run_multi_gpu, parse_gpus


def reasoner_prompt(representation: str, question: str) -> str:
    """Verbatim TextFlow reasoner prompt."""
    return f"{representation}\n\nQuestion: {question}\nAnswer:"


def _load_gt_mermaid_flowvqa(gt_dict: dict, key: str) -> str:
    """For FlowVQA: 'mermaid' field is in test.json sample dict."""
    return gt_dict.get(key, {}).get("mermaid", "")


def _load_gt_mermaid_flowlearn(gt_dict: dict, key: str, gt_mermaid_dir: str) -> str:
    """For FlowLearn: read mermaid_word/txt/<key>.txt."""
    p = os.path.join(gt_mermaid_dir, f"{key}.txt")
    if not os.path.exists(p):
        return ""
    with open(p) as f:
        return f.read().strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["flowvqa", "flowlearn"])
    p.add_argument("--test-json", required=True)
    p.add_argument("--gt-mermaid-dir", default="",
                   help="For flowlearn: dir with <key>.txt mermaid sources. "
                        "For flowvqa: ignored (read from test.json 'mermaid' field).")
    p.add_argument("--output-file", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--reasoner-max-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.test_json) as f:
        gt = json.load(f)
    keys = sorted(gt.keys())
    if args.limit > 0:
        keys = keys[: args.limit]

    existing: dict = {}
    if args.resume and os.path.exists(args.output_file):
        try:
            with open(args.output_file) as f:
                existing = json.load(f)
            print(f"[resume] loaded {len(existing)} prior images")
        except Exception:
            existing = {}

    tasks = []
    skipped = 0
    n_no_mermaid = 0
    for k in keys:
        if args.dataset == "flowvqa":
            gz = _load_gt_mermaid_flowvqa(gt, k)
        else:
            gz = _load_gt_mermaid_flowlearn(gt, k, args.gt_mermaid_dir)
        if not gz:
            n_no_mermaid += 1
            continue

        prior = existing.get(k, {})
        qa_dict = gt[k].get("qa", {})
        for qid, item in qa_dict.items():
            if qid in prior and prior[qid].get("predicted"):
                skipped += 1
                continue
            tasks.append({
                "image_path": "",
                "prompt": reasoner_prompt(gz, item["Q"]),
                "out_key": (k, qid),
            })

    print(f"[plan] images missing mermaid: {n_no_mermaid}/{len(keys)}")
    print(f"[plan] tasks={len(tasks)}  skipped={skipped}")
    if not tasks:
        return

    gpus = parse_gpus(args.gpus)
    results = run_multi_gpu(
        tasks,
        gpus=gpus,
        base_model_path=args.base_model_path,
        adapter_path="",
        max_new_tokens=args.reasoner_max_tokens,
        system_prompt="",
        no_image=True,  # text-only
    )

    out: dict = existing
    for r in results:
        k, qid = r["out_key"]
        if k not in out:
            out[k] = {}
        gt_item = gt[k]["qa"][qid]
        out[k][qid] = {
            "Q": gt_item["Q"],
            "A1": gt_item.get("A1", ""),
            "A2": gt_item.get("A2", ""),
            "A3": gt_item.get("A3", ""),
            "type": gt_item.get("type", ""),
            "predicted": r["predicted"],
            "ok": r["ok"],
        }

    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] saved → {args.output_file}  "
          f"({sum(len(v) for v in out.values())} predictions)")


if __name__ == "__main__":
    main()
