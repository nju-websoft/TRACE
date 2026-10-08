#!/bin/bash
# ============================================================
# train_sam3.sh
#
# Train the SAM3 arrowhead detection model.
# Each dataset trains its own independent detection model.
#
# Usage (run from anywhere; repo root is auto-detected from the script location):
# bash scripts/train_sam3.sh [--gpus N] [--datasets "fcb flowlearn flowvqa"]
#
# Arguments:
# --gpus N Number of GPUs to use (default 6, overridden by --gpu-ids)
# --gpu-ids "2,3,4,5" Which GPU ids to use (GPU count is derived automatically)
# --datasets "d1 d2" List of datasets to train
# --max-epochs N Max number of training epochs (default 30)
#
# Examples:
# bash scripts/train_sam3.sh --gpu-ids "0,1,2,3"
# bash scripts/train_sam3.sh --gpu-ids "4,5,6,7" --datasets "bpmn flowgen_easy"
# ============================================================

set -e

# Repo root is derived from this script's location (scripts/ -> repo root),
# so the script is portable and needs no hard-coded absolute paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DETECTION_DIR="${REPO_ROOT}/detection"
DATASETS_DIR="${DETECTION_DIR}/datasets"
CONFIG_DIR="${REPO_ROOT}/sam3/train/configs"
BASE_CONFIG="${CONFIG_DIR}/arrowhead_detection.yaml"

# Default parameters
NUM_GPUS=6
GPU_IDS="2,3,4,5,6,7"   # e.g. "2,3,4,5" to pin specific ids; empty -> use the first NUM_GPUS GPUs
DATASETS=( "flowgen_easy" "flowgen_medium" "flowgen_hard")
MAX_EPOCHS=30

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            NUM_GPUS="$2"; shift 2 ;;
        --gpu-ids)
            GPU_IDS="$2"; shift 2 ;;
        --datasets)
            IFS=' ' read -r -a DATASETS <<< "$2"; shift 2 ;;
        --max-epochs)
            MAX_EPOCHS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Select GPUs
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    # Derive the actual GPU count from the pinned ids
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  SAM3 Arrowhead Detection — Per-Dataset Training${NC}"
echo -e "${BLUE}  GPUs:       ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})${NC}"
echo -e "${BLUE}  Datasets:   ${DATASETS[*]}${NC}"
echo -e "${BLUE}  Max Epochs: ${MAX_EPOCHS}${NC}"
echo -e "${BLUE}  Start Time: $(date)${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

TOTAL=${#DATASETS[@]}
SUCCESS=0
FAILED=0

for ds in "${DATASETS[@]}"; do
    DS_DIR="${DATASETS_DIR}/sam3_${ds}"
    TRAIN_ANN="${DS_DIR}/train.json"
    VAL_ANN="${DS_DIR}/val.json"

    # Check that the dataset exists
    if [ ! -f "$TRAIN_ANN" ] || [ ! -f "$VAL_ANN" ]; then
        echo -e "${RED}  ✗ Dataset ${ds} not found: ${DS_DIR}${NC}"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Output directory
    EXP_DIR="${DETECTION_DIR}/runs/arrowhead_${ds}"
    mkdir -p "${EXP_DIR}"

    LOG_FILE="${EXP_DIR}/train_${TIMESTAMP}.log"

    echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
    echo -e "${GREEN}  ▶ [$((SUCCESS + FAILED + 1))/${TOTAL}] Training dataset: ${ds}${NC}"
    echo -e "    train_ann: ${TRAIN_ANN}"
    echo -e "    val_ann:   ${VAL_ANN}"
    echo -e "    output:    ${EXP_DIR}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"

    # Skip already-finished training (enough checkpoints present)
    CKPT_DIR="${EXP_DIR}/checkpoints"
    if [ -d "$CKPT_DIR" ]; then
        CKPT_COUNT=$(ls "$CKPT_DIR" 2>/dev/null | grep -c "checkpoint_" || true)
        if [ "$CKPT_COUNT" -ge 3 ]; then
            echo -e "    Already has ${CKPT_COUNT} checkpoints, skipping training"
            SUCCESS=$((SUCCESS + 1))
            continue
        fi
    fi

    # Generate a dataset-specific config from the base config (substitute paths and params).
    # Paths injected here are RELATIVE to the repo root (we cd into REPO_ROOT before launching).
    REL_DS="detection/datasets/sam3_${ds}"
    REL_LOG="detection/runs/arrowhead_${ds}"
    DS_CONFIG="${CONFIG_DIR}/arrowhead_${ds}.yaml"
    sed \
        -e "s|experiment_log_dir:.*|experiment_log_dir: ${REL_LOG}|" \
        -e "s|train_ann:.*|train_ann: ${REL_DS}/train.json|" \
        -e "s|val_ann:.*|val_ann: ${REL_DS}/val.json|" \
        -e "s|img_folder:.*|img_folder: ${REL_DS}|" \
        -e "s|max_epochs:.*|max_epochs: ${MAX_EPOCHS}|" \
        -e "s|gpus_per_node:.*|gpus_per_node: ${NUM_GPUS}|" \
        "$BASE_CONFIG" > "$DS_CONFIG"

    echo -e "    Generated config: ${DS_CONFIG}"

    cd "${REPO_ROOT}"

    python sam3/train/train.py \
        -c "configs/arrowhead_${ds}" \
        --use-cluster 0 \
        --num-gpus "${NUM_GPUS}" \
        --num-nodes 1 \
        2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}  ✅ ${ds} training finished${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}  ❌ ${ds} training failed (exit=${EXIT_CODE})${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo -e "    Log: ${LOG_FILE}"
    echo -e "    Checkpoints: ${EXP_DIR}/checkpoints/"
    echo ""

done

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Training Summary${NC}"
echo -e "${BLUE}  Total: ${TOTAL}  Success: ${SUCCESS}  Failed: ${FAILED}${NC}"
echo -e "${BLUE}  End Time: $(date)${NC}"
echo -e "${BLUE}============================================================${NC}"

# List all training results
echo ""
echo "Training result directories:"
for ds in "${DATASETS[@]}"; do
    EXP_DIR="${DETECTION_DIR}/runs/arrowhead_${ds}"
    if [ -d "${EXP_DIR}/checkpoints" ]; then
        CKPT_COUNT=$(ls "${EXP_DIR}/checkpoints/" 2>/dev/null | wc -l)
        echo "  ${ds}: ${EXP_DIR}/checkpoints/ (${CKPT_COUNT} files)"
    fi
done

[ $FAILED -gt 0 ] && exit 1 || exit 0
