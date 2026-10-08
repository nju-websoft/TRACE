#!/bin/bash
# ============================================================
# train_yolo.sh
#
# Sequentially train YOLO arrowhead detection models (one run per dataset).
# Mirrors scripts/train_sam3.sh and keeps the config aligned:
# - max_epochs default 30
# - 6 GPUs (same default as SAM3), override with --gpu-ids
# - output: detection/yolo/output/<ds>/ (weights/best.pt, etc.)
#
# Usage:
# bash scripts/train_yolo.sh
# bash scripts/train_yolo.sh --gpu-ids "0,1" --datasets "bpmn fcb"
# bash scripts/train_yolo.sh --model yolo11l.pt --max-epochs 50
#
# Arguments:
# --gpu-ids "0,1,2,3" Which GPUs to use (default 2,3,4,5,6,7)
# --datasets "d1 d2" Datasets to train (default: all 7)
# --max-epochs N Training epochs (default 30)
# --model PATH YOLO pretrained weights (default yolo11l.pt)
# --imgsz N Input image size (default 1280, arrowheads are small)
# --batch N Batch size (default -1, auto)
# --skip-convert Skip the COCO->YOLO data conversion
# --force-convert Force-redo the conversion (overwrite existing symlinks/labels)
# ============================================================
set -e

# Activate the conda env that has ultralytics (the sam3 env has ultralytics 8.4.43)
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate "<CONDA_ENV>"

# Repo root is derived from this script's location (scripts/ -> repo root),
# so the script is portable and needs no hard-coded absolute paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DETECTION_DIR="${REPO_ROOT}/detection"
YOLO_DIR="${DETECTION_DIR}/yolo"
DATA_ROOT="${YOLO_DIR}/data"
OUTPUT_ROOT="${YOLO_DIR}/output"
CONVERT_PY="${YOLO_DIR}/scripts/convert_coco_to_yolo.py"

# Default parameters
GPU_IDS="2,3,4,5,6,7"
DATASETS=("flowvqa" "flowgen_easy" "flowgen_medium" "flowgen_hard")
MAX_EPOCHS=30
MODEL="${YOLO_DIR}/pretrained/yolo11x.pt"   # favors accuracy; drop to yolo11l.pt if VRAM is tight
SKIP_CONVERT=0
FORCE_CONVERT=0

# Run several bs_per_device trials per dataset (1/2/4) -> output/<ds>_bs<global>/
# Global batch = bs_per_device x num_gpus
BS_PER_DEVICE_LIST=(2 4)

# ============================================================
# Per-dataset imgsz / batch
# imgsz rule of thumb: ~ median long edge of the dataset images; too large adds
# interpolation artifacts and hurts accuracy
# batch: larger imgsz -> smaller batch (1 image/GPU on 4 GPUs is the floor)
#
# Measured: bpmn imgsz=1280 -> mAP50 0.99, imgsz=2048 -> mAP50 0.95 (5pp drop)
#
# Reference median long edge per dataset:
# bpmn=1199, fcb=1333, flowlearn=470,
# flowvqa=2533, flowgen_easy=4303, flowgen_medium=1507, flowgen_hard=3288
# ============================================================
declare -A IMGSZ_FOR
IMGSZ_FOR=(
    ["bpmn"]=1280
    ["fcb"]=1280
    ["flowlearn"]=640          # source images are tiny (~470); 640 is enough without over-upscaling
    ["flowvqa"]=2048
    ["flowgen_easy"]=2048
    ["flowgen_medium"]=1920    # median 1507, 1920 leaves some headroom
    ["flowgen_hard"]=2048
)

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

# --imgsz is a global override: if passed, every dataset uses this value
GLOBAL_IMGSZ=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-ids)         GPU_IDS="$2"; shift 2 ;;
        --datasets)        IFS=' ' read -r -a DATASETS <<< "$2"; shift 2 ;;
        --max-epochs)      MAX_EPOCHS="$2"; shift 2 ;;
        --model)           MODEL="$2"; shift 2 ;;
        --imgsz)           GLOBAL_IMGSZ="$2"; shift 2 ;;
        --bs-per-device)   IFS=' ' read -r -a BS_PER_DEVICE_LIST <<< "$2"; shift 2 ;;
        --skip-convert)    SKIP_CONVERT=1; shift ;;
        --force-convert)   FORCE_CONVERT=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# NOTE: do NOT export CUDA_VISIBLE_DEVICES!
# ultralytics 8.4.x select_device() overrides this env var with the user-passed
# device string, interpreting "5" as "physical GPU 0" (under CUDA_VISIBLE_DEVICES=5,
# cuda:0 is physical 5). After the rewrite CUDA_VISIBLE_DEVICES=0 -> actually runs
# physical GPU 0 -> collides with others' training -> OOM.
# Fix: pass the absolute GPU ids straight to yolo and rely on its own select_device.
unset CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
DEVICE_LIST="$GPU_IDS"   # pass absolute ids straight to yolo

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${DETECTION_DIR}/training_logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
MAIN_LOG="${LOG_DIR}/yolo_${TIMESTAMP}.log"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}  YOLO Arrowhead Detection — Per-Dataset Training${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}  Model:      ${MODEL}${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}  GPUs:       ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${GPU_IDS})${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}  Datasets:   ${DATASETS[*]}${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}  Max Epochs: ${MAX_EPOCHS}${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}  BS_PER_DEVICE_LIST: ${BS_PER_DEVICE_LIST[*]}${NC}" | tee -a "$MAIN_LOG"
if [ -n "$GLOBAL_IMGSZ" ]; then
    echo -e "${BLUE}  Override:   imgsz=${GLOBAL_IMGSZ}${NC}" | tee -a "$MAIN_LOG"
fi
echo -e "${BLUE}  Trials (each ds × each bs_per_device):${NC}" | tee -a "$MAIN_LOG"
for _ds in "${DATASETS[@]}"; do
    _is="${GLOBAL_IMGSZ:-${IMGSZ_FOR[$_ds]:-1280}}"
    for _bspd in "${BS_PER_DEVICE_LIST[@]}"; do
        _gbs=$((_bspd * NUM_GPUS))
        echo -e "${BLUE}    ${_ds}_bs${_gbs}: imgsz=${_is}, bs_per_device=${_bspd}, global_batch=${_gbs}${NC}" | tee -a "$MAIN_LOG"
    done
done
echo -e "${BLUE}  Start Time: $(date)${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# ============================================================
# Stage 0: COCO -> YOLO data conversion
# ============================================================
if [ "$SKIP_CONVERT" -eq 0 ]; then
    echo -e "${CYAN}[Stage 0] COCO -> YOLO data conversion${NC}" | tee -a "$MAIN_LOG"
    CONVERT_ARGS=(--datasets "${DATASETS[@]}")
    [ "$FORCE_CONVERT" -eq 1 ] && CONVERT_ARGS+=(--force)
    python3 "$CONVERT_PY" "${CONVERT_ARGS[@]}" 2>&1 | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
fi

# ============================================================
# Stage 1: sequential training (one run per ds x per bs_per_device)
# ============================================================
TOTAL=$((${#DATASETS[@]} * ${#BS_PER_DEVICE_LIST[@]}))
SUCCESS=0
FAILED=0

for ds in "${DATASETS[@]}"; do
    DATA_YAML="${DATA_ROOT}/${ds}/data.yaml"

    if [ ! -f "$DATA_YAML" ]; then
        echo -e "${RED}  ✗ missing data.yaml: ${DATA_YAML}${NC}" | tee -a "$MAIN_LOG"
        FAILED=$((FAILED + ${#BS_PER_DEVICE_LIST[@]}))
        continue
    fi

    # imgsz: --imgsz override > dict > 1280
    DS_IMGSZ="${GLOBAL_IMGSZ:-${IMGSZ_FOR[$ds]:-1280}}"

    for bspd in "${BS_PER_DEVICE_LIST[@]}"; do
        DS_BATCH=$((bspd * NUM_GPUS))
        RUN_NAME="${ds}_bs${DS_BATCH}"
        EXP_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
        LOG_FILE="${LOG_DIR}/yolo_${RUN_NAME}_${TIMESTAMP}.log"

        echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}" | tee -a "$MAIN_LOG"
        echo -e "${GREEN}  ▶ [$((SUCCESS + FAILED + 1))/${TOTAL}] training ${RUN_NAME}  (imgsz=${DS_IMGSZ}, bs_per_device=${bspd}, global_batch=${DS_BATCH})${NC}" | tee -a "$MAIN_LOG"
        echo -e "    data:   ${DATA_YAML}" | tee -a "$MAIN_LOG"
        echo -e "    output: ${EXP_DIR}" | tee -a "$MAIN_LOG"
        echo -e "    log:    ${LOG_FILE}" | tee -a "$MAIN_LOG"

        # resume: skip if best.pt already exists
        if [ -f "${EXP_DIR}/weights/best.pt" ]; then
            echo -e "    ${YELLOW}weights/best.pt already exists, skipping${NC}" | tee -a "$MAIN_LOG"
            SUCCESS=$((SUCCESS + 1))
            continue
        fi

        # Train via the ultralytics CLI
        yolo detect train \
            model="${MODEL}" \
            data="${DATA_YAML}" \
            epochs="${MAX_EPOCHS}" \
            imgsz="${DS_IMGSZ}" \
            batch="${DS_BATCH}" \
            device="${DEVICE_LIST}" \
            project="${OUTPUT_ROOT}" \
            name="${RUN_NAME}" \
            exist_ok=True \
            save=True \
            save_period=5 \
            patience=99999 \
            workers=8 \
            2>&1 | tee "$LOG_FILE" | tee -a "$MAIN_LOG"

        EC=${PIPESTATUS[0]}
        if [ $EC -eq 0 ]; then
            echo -e "${GREEN}  ✅ ${RUN_NAME} training finished${NC}" | tee -a "$MAIN_LOG"
            SUCCESS=$((SUCCESS + 1))
        else
            echo -e "${RED}  ❌ ${RUN_NAME} training failed (exit=${EC})${NC}" | tee -a "$MAIN_LOG"
            FAILED=$((FAILED + 1))
        fi
        echo "" | tee -a "$MAIN_LOG"
    done
done

# ============================================================
# Summary
# ============================================================
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}  YOLO training finished${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}  total=${TOTAL}  success=${SUCCESS}  failed=${FAILED}${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}  End Time: $(date)${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}" | tee -a "$MAIN_LOG"

echo "" | tee -a "$MAIN_LOG"
echo "Training results:" | tee -a "$MAIN_LOG"
for ds in "${DATASETS[@]}"; do
    for bspd in "${BS_PER_DEVICE_LIST[@]}"; do
        DS_BATCH=$((bspd * NUM_GPUS))
        BEST="${OUTPUT_ROOT}/${ds}_bs${DS_BATCH}/weights/best.pt"
        if [ -f "$BEST" ]; then
            SZ=$(du -h "$BEST" | cut -f1)
            echo "  ${ds}_bs${DS_BATCH}: ${BEST} (${SZ})" | tee -a "$MAIN_LOG"
        else
            echo "  ${ds}_bs${DS_BATCH}: (no best.pt)" | tee -a "$MAIN_LOG"
        fi
    done
done

echo "" | tee -a "$MAIN_LOG"
echo "  main log: ${MAIN_LOG}"

[ $FAILED -gt 0 ] && exit 1 || exit 0
