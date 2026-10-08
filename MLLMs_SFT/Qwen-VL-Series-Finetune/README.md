# Qwen-VL Fine-tuning (TRACE)

This directory holds the Qwen-VL LoRA fine-tuning code used by TRACE, adapted from an open-source
Qwen-VL fine-tuning framework (HuggingFace + PEFT + DeepSpeed ZeRO-3).

The TRACE-specific entry points live in [`scripts/`](scripts/) — see the repository root
[`README.md`](../../README.md) for the full pipeline:

| Script | Purpose |
|--------|---------|
| `scripts/train_all_TRACE_grid_search.sh` | per-arrow (TRACE) path: LoRA fine-tune → pick best checkpoint → test inference → score |
| `scripts/train_all_E2E_grid_search.sh`   | whole-image (E2E) path |
| `scripts/train_leave_one_out_TRACE.sh` / `train_leave_one_out_E2E.sh` | cross-style leave-one-out |
| `scripts/eval_baseline_no_sft.sh`        | base-model (no-SFT) baseline |
| `scripts/train_early_stop.sh`            | a single training run with early stopping |

Training uses DeepSpeed ZeRO-3 (`scripts/zero3.json`). LoRA settings (rank, alpha, dropout, learning
rate), the batch-size grid, GPU ids, and model/data paths are configured inside those scripts.
Replace the `<DATA_ROOT>` / `<MODEL_ROOT>` / `<OUTPUT_ROOT>` / `<CONDA_ENV>` placeholders with your own
values before running.
