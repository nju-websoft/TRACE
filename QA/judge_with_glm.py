"""LLM-as-judge for FlowVQA / FlowLearn predictions using GLM-4.6V via Z.ai API
(OpenAI-compatible).

Builds TextFlow's judge prompt (question + predicted + ground-truth) and asks
GLM-4.6V to output "Correct" or "Incorrect". For FlowVQA we score against the
best of A1/A2/A3 (any match → correct), matching TextFlow's protocol.

Outputs incrementally (--save-every N), so you can monitor progress by tail'ing
the output file size or by reading it directly.

Usage:
    ZAI_API_KEY=... python QA/judge_with_glm.py \\
        --prediction-file QA/results/textflow_subset/trace_tools_flowvqa.json \\
        --output-file     QA/results/textflow_subset/trace_tools_flowvqa_judged.json \\
        --model           glm-4.6v \\
        --concurrency 10 --n-votes 1
"""

import os
import sys
import json
import asyncio
import argparse
from typing import Dict, List

from openai import AsyncOpenAI


JUDGE_PROMPT_TMPL = """Task: Verify if the provided answer is correct based on the given ground truth.

You are given a question, an answer and the ground truth. Your task is to determine whether the provided answer matches the ground truth. Output "Correct" if the answer matches groud truth, otherwise output "Incorrect".

Question: {q}

Answer: {a}

Ground Truth: {gt}"""


def _normalize_item(item: dict) -> dict:
    q = item.get("Q") or item.get("question") or ""
    pred = item.get("predicted") or item.get("response") or ""
    labels = []
    for k in ("A1", "A2", "A3"):
        v = item.get(k)
        if v and isinstance(v, str):
            labels.append(v)
    if not labels and item.get("label"):
        labels = [item["label"]]
    if not labels and item.get("gt"):
        labels = [item["gt"]]
    return {"q": q, "pred": pred, "labels": [l for l in labels if l]}


async def _judge_one_call(
    client: AsyncOpenAI, model: str, prompt: str,
    temperature: float, max_retries: int,
) -> int:
    """Single judge call → 1 (Correct) or 0 (Incorrect/unknown)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )
            text = (resp.choices[0].message.content or "").strip().lower()
            if "incorrect" in text:
                return 0
            if "correct" in text:
                return 1
            return 0
        except Exception as e:
            last_err = e
            await asyncio.sleep(min(2 ** attempt, 10))
    print(f"  [judge-err] gave up after {max_retries} retries: {last_err}", flush=True)
    return 0


async def _judge_one_item(
    sem: asyncio.Semaphore, client: AsyncOpenAI, model: str,
    item: dict, n_votes: int, temperature: float, max_retries: int,
) -> dict:
    n = _normalize_item(item)
    if not n["pred"] or not n["labels"]:
        return {"votes": [0] * n_votes, "correct": 0,
                "skipped_reason": "no pred or no label"}

    async def _vote_for_label(label):
        prompt = JUDGE_PROMPT_TMPL.format(q=n["q"], a=n["pred"], gt=label)
        async with sem:
            votes = []
            for _ in range(n_votes):
                v = await _judge_one_call(client, model, prompt, temperature, max_retries)
                votes.append(v)
        return votes

    label_votes_lists = await asyncio.gather(
        *[_vote_for_label(l) for l in n["labels"]]
    )
    best = max(label_votes_lists, key=lambda v: sum(v))
    final = 1 if sum(best) >= (n_votes + 1) // 2 else 0
    return {"votes": best, "correct": final}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prediction-file", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--model", default="glm-4.6v")
    p.add_argument("--api-base", default="https://api.z.ai/api/paas/v4/")
    p.add_argument("--api-key", default="")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--n-votes", type=int, default=1,
                   help="GLM-4.6V at temperature=0 is deterministic, so 1 vote is enough.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True)
    return p.parse_args()


async def main_async(args):
    api_key = (args.api_key
               or os.environ.get("ZAI_API_KEY")
               or os.environ.get("ZHIPU_API_KEY"))
    if not api_key:
        raise SystemExit("ZAI_API_KEY / ZHIPU_API_KEY not set, and --api-key not given")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.prediction_file) as f:
        data = json.load(f)

    existing: dict = {}
    if args.resume and os.path.exists(args.output_file):
        try:
            with open(args.output_file) as f:
                existing = json.load(f)
            done = sum(1 for k in existing for qid in existing[k]
                       if "correct" in existing[k][qid])
            print(f"[resume] {done} predictions already judged", flush=True)
        except Exception as e:
            print(f"[resume] could not load existing: {e}")
            existing = {}

    tasks: List = []
    keys = sorted(data.keys())
    if args.limit > 0:
        keys = keys[: args.limit]
    for k in keys:
        for qid, item in data[k].items():
            if "correct" in existing.get(k, {}).get(qid, {}):
                continue
            tasks.append((k, qid, item))

    if not tasks:
        print("[plan] nothing to judge.")
        return

    print(f"[plan] judge tasks: {len(tasks)}  "
          f"concurrency={args.concurrency}  n_votes={args.n_votes}", flush=True)

    client = AsyncOpenAI(api_key=api_key, base_url=args.api_base)
    sem = asyncio.Semaphore(args.concurrency)

    out: dict = existing
    done = 0
    n_correct = 0
    n_total_judged = sum(1 for k in out for qid in out[k]
                          if out[k][qid].get("correct") == 1)
    n_pred_judged = sum(1 for k in out for qid in out[k] if "correct" in out[k][qid])

    coros = [
        _judge_one_item(sem, client, args.model, item,
                         args.n_votes, args.temperature, args.max_retries)
        for (_, _, item) in tasks
    ]

    for fut in asyncio.as_completed(
        [_pack(i, c) for i, c in enumerate(coros)]
    ):
        idx, r = await fut
        k, qid, item = tasks[idx]
        if k not in out:
            out[k] = {}
        if qid not in out[k]:
            out[k][qid] = dict(item)
        out[k][qid]["votes"] = r["votes"]
        out[k][qid]["correct"] = r["correct"]
        if "skipped_reason" in r:
            out[k][qid]["skipped_reason"] = r["skipped_reason"]

        done += 1
        n_pred_judged += 1
        if r["correct"] == 1:
            n_total_judged += 1

        if done % args.save_every == 0 or done == len(tasks):
            with open(args.output_file, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            acc = n_total_judged / max(n_pred_judged, 1)
            print(f"  [judge {done}/{len(tasks)}] saved. "
                  f"running_acc={n_total_judged}/{n_pred_judged}={acc:.4f}",
                  flush=True)

    n_total = sum(len(v) for v in out.values())
    n_correct_final = sum(1 for v in out.values() for x in v.values()
                           if x.get("correct") == 1)
    print(f"\n[FINAL] {n_correct_final}/{n_total} = {n_correct_final/n_total:.4f}",
          flush=True)


async def _pack(i, c):
    r = await c
    return i, r


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
