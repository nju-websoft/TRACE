#!/bin/bash
# Compare trained (LoRA) vs untrained Qwen3-VL-4B on FlowVQA + FlowLearn QA.
# Usage:
#   bash QA/run_compare.sh                # full run
#   LIMIT=50 bash QA/run_compare.sh       # quick sanity check (50 images per dataset)
#   GPUS="4,5,6,7" bash QA/run_compare.sh

set -e

REPO_ROOT="<REPO_ROOT>"
cd "$REPO_ROOT"

# Activate the conda env that has torch + transformers + peft (<CONDA_ENV>).
CONDA_ENV="${CONDA_ENV:-<CONDA_ENV>}"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "<CONDA_ROOT>")"
# shellcheck disable=SC1090,SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "  Python: $(which python)"
python -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

# ---- config ----
BASE_MODEL="${BASE_MODEL:-<MODEL_ROOT>/Qwen3-VL-4B-Instruct}"
ADAPTER_FLOWVQA="${ADAPTER_FLOWVQA:-${REPO_ROOT}/MLLMs_SFT/Qwen-VL-Series-Finetune/output/qwen3_vl_4b_arrow_flowvqa_bs64/checkpoint-834}"
ADAPTER_FLOWLEARN="${ADAPTER_FLOWLEARN:-${REPO_ROOT}/MLLMs_SFT/Qwen-VL-Series-Finetune/output/qwen3_vl_4b_arrow_flowlearn_bs128/checkpoint-1185}"
# Pre-extracted LoRA-triplet dirs (one image-stem subdir each, with arrow_triplets.json)
TRIPLETS_FLOWVQA="${TRIPLETS_FLOWVQA:-${REPO_ROOT}/qwen3_vl_4b_arrow_flowvqa_bs64_best_sam3_ft}"
TRIPLETS_FLOWLEARN="${TRIPLETS_FLOWLEARN:-${REPO_ROOT}/qwen3_vl_4b_arrow_flowlearn_bs128_best_sam3_ft}"
# Variant set: "base lora triplets" — drop any to skip
VARIANTS="${VARIANTS:-base lora triplets}"
GPUS="${GPUS:-0,1,2,3}"
LIMIT="${LIMIT:-0}"   # 0 = all

OUT_DIR="${REPO_ROOT}/QA/results"
mkdir -p "$OUT_DIR"

# Sanity: check the base model and adapters exist
for p in "$BASE_MODEL" "$ADAPTER_FLOWVQA" "$ADAPTER_FLOWLEARN"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: path not found: $p" >&2
        exit 1
    fi
done
# Triplets dirs are optional but warn if missing when 'triplets' variant requested
if [[ " $VARIANTS " == *" triplets "* ]]; then
    for p in "$TRIPLETS_FLOWVQA" "$TRIPLETS_FLOWLEARN"; do
        if [ ! -e "$p" ]; then
            echo "WARN: triplets dir not found: $p" >&2
        fi
    done
fi

LIMIT_FLAG=""
[ "$LIMIT" -gt 0 ] && LIMIT_FLAG="--limit $LIMIT"

echo "================================================================"
echo "  base model        : $BASE_MODEL"
echo "  flowvqa adapter   : $ADAPTER_FLOWVQA"
echo "  flowlearn adapter : $ADAPTER_FLOWLEARN"
echo "  flowvqa triplets   : $TRIPLETS_FLOWVQA"
echo "  flowlearn triplets : $TRIPLETS_FLOWLEARN"
echo "  variants          : $VARIANTS"
echo "  GPUs              : $GPUS"
echo "  LIMIT             : ${LIMIT:-all}"
echo "================================================================"

run_pair() {
    # $1 dataset name (flowvqa|flowlearn)
    # $2 adapter path
    # $3 triplets dir
    local DS="$1"
    local ADAPTER="$2"
    local TRIPLETS="$3"

    local INFER_PY="QA/infer_${DS}.py"

    for variant in $VARIANTS; do
        local out_pred="${OUT_DIR}/${DS}_qwen3_4b_${variant}.json"
        local extra_arg=""
        case "$variant" in
            base)    ;;
            lora)    extra_arg="--adapter-path $ADAPTER" ;;
            triplets) extra_arg="--triplets-dir $TRIPLETS" ;;
            *) echo "unknown variant: $variant" >&2; continue ;;
        esac
        echo
        echo "==== ${DS} / ${variant} → $out_pred ===="
        python "$INFER_PY" \
            --output-file "$out_pred" \
            --base-model-path "$BASE_MODEL" \
            $extra_arg \
            --gpus "$GPUS" \
            $LIMIT_FLAG \
            --resume

        echo
        echo "---- score ${DS} / ${variant} ----"
        if [ "$DS" = "flowvqa" ]; then
            # FlowVQA: GLM-as-judge on the TextFlow subset (set ZAI_API_KEY)
            python QA/judge_with_glm.py \
                --prediction-file "$out_pred" \
                --output-file "${out_pred%.json}_judged.json" \
                --api-key "${ZAI_API_KEY:-}"
        else
            # FlowLearn: deterministic string scoring
            python QA/rate_flowlearn.py --prediction-file "$out_pred"
        fi
    done
}

run_pair flowvqa  "$ADAPTER_FLOWVQA"  "$TRIPLETS_FLOWVQA"
run_pair flowlearn "$ADAPTER_FLOWLEARN" "$TRIPLETS_FLOWLEARN"

echo
echo "All done. Results in $OUT_DIR"
