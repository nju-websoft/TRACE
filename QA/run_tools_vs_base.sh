#!/usr/bin/env bash
# Full-set comparison: Qwen3-VL-4B base vs base+tool-calling on FlowVQA + FlowLearn.
#   triplets sources are the user's pre-extracted LoRA outputs.
set -euo pipefail

source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate "<CONDA_ENV>"

cd "<REPO_ROOT>"

BASE_MODEL="<MODEL_ROOT>/Qwen3-VL-4B-Instruct"
GPUS=0,1,2,3,4,5,6
OUTDIR=QA/results/full
mkdir -p "$OUTDIR" QA/logs

VQA_TRIPLETS="<OUTPUT_ROOT>/qwen3_vl_4b_arrow_flowvqa_bs64_best_sam3_ft"
FL_TRIPLETS="<OUTPUT_ROOT>/qwen3_vl_4b_arrow_flowlearn_bs128_best_sam3_ft"

ts=$(date +%Y%m%d_%H%M%S)
LOG=QA/logs/tools_vs_base_${ts}.log
echo "log → $LOG"

run_step() {
  local label="$1"; shift
  echo
  echo "================================================================"
  echo "[${label}] $(date)"
  echo "================================================================"
  "$@"
}

{
  # ---------------- FlowVQA ----------------
  run_step "flowvqa-base" \
    python QA/infer_flowvqa.py \
      --output-file "$OUTDIR/flowvqa_q3_4b_base.json" \
      --base-model-path "$BASE_MODEL" \
      --gpus "$GPUS"

  run_step "flowvqa-tools" \
    python QA/infer_flowvqa_tools.py \
      --output-file "$OUTDIR/flowvqa_q3_4b_tools.json" \
      --triplets-dir "$VQA_TRIPLETS" \
      --base-model-path "$BASE_MODEL" \
      --gpus "$GPUS"

  # ---------------- FlowLearn ----------------
  run_step "flowlearn-base" \
    python QA/infer_flowlearn.py \
      --output-file "$OUTDIR/flowlearn_q3_4b_base.json" \
      --base-model-path "$BASE_MODEL" \
      --gpus "$GPUS"

  run_step "flowlearn-tools" \
    python QA/infer_flowlearn_tools.py \
      --output-file "$OUTDIR/flowlearn_q3_4b_tools.json" \
      --triplets-dir "$FL_TRIPLETS" \
      --base-model-path "$BASE_MODEL" \
      --gpus "$GPUS"

  # ---------------- Score ----------------
  echo
  echo "================================================================"
  echo "[score] $(date)"
  echo "================================================================"
  # FlowVQA: GLM-as-judge, scored on the TextFlow subset (set ZAI_API_KEY).
  # FlowLearn: deterministic string scoring.
  python QA/judge_with_glm.py  --prediction-file "$OUTDIR/flowvqa_q3_4b_base.json"  --output-file "$OUTDIR/flowvqa_q3_4b_base_judged.json"  --api-key "${ZAI_API_KEY:-}"
  python QA/judge_with_glm.py  --prediction-file "$OUTDIR/flowvqa_q3_4b_tools.json" --output-file "$OUTDIR/flowvqa_q3_4b_tools_judged.json" --api-key "${ZAI_API_KEY:-}"
  python QA/rate_flowlearn.py  --prediction-file "$OUTDIR/flowlearn_q3_4b_base.json"
  python QA/rate_flowlearn.py  --prediction-file "$OUTDIR/flowlearn_q3_4b_tools.json"

  echo
  echo "[done] $(date)"
} 2>&1 | tee "$LOG"
