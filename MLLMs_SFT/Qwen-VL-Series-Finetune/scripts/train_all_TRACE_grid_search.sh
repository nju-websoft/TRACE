#!/bin/bash

# ============================================================
# train_all_TRACE_grid_search.sh
#
# Arrow version Grid Search training script
#
# vs triplet version difference:
# - training data as arrow format ({dataset}_arrow_training_all.json)
# - cbd / fcb / bpmn have official val, use pregenerated val arrow data, not from train split
# - other datasets (fca, flowlearn, flowvqa, flowgen) from train split 90/10
#
# Flow:
# 1. prepare val data (cbd/fcb/bpmn use official val, other from train split)
# 2. Grid Search over Batch Size + early stopping (patience=1)
# 3. across BS pick best checkpoint → test inference → scoring
#
# Usage:
# cd MLLMs_SFT/Qwen-VL-Series-Finetune
# bash scripts/train_all_TRACE_grid_search.sh
#
# Note: cbd/fcb/bpmn val arrow data must be generated beforehand:
# cd "<REPO_ROOT>"
# python -m gen_data.prepare_training_data --datasets all
# ============================================================

SCRIPT_CWD="$(pwd)"
echo $SCRIPT_CWD

# ============================================================
# command-line arg parsing (eval stage)
# --arrow-sources "sam sam3_ft yolo groundtruth" multiple source loop
# --arrow-source <src> single source (override --arrow-sources)
# --skip-eval skip eval Stage, onlytraining
# ============================================================
ARROW_SOURCE=""
ARROW_SOURCES="groundtruth"
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arrow-source)   ARROW_SOURCE="$2"; shift 2 ;;
        --arrow-sources)  ARROW_SOURCES="$2"; shift 2 ;;
        --skip-eval)      SKIP_EVAL=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[ -n "$ARROW_SOURCE" ] && ARROW_SOURCES="$ARROW_SOURCE"
for _src in $ARROW_SOURCES; do
    if [[ "$_src" != "sam" && "$_src" != "sam3_ft" && "$_src" != "yolo" && "$_src" != "groundtruth" ]]; then
        echo "Error: invalid arrow source '$_src'"; exit 1
    fi
done

# ============================================================
# SAM3 / YOLO detector: by dataset auto-select best checkpoint
# ============================================================
DETECTION_RUNS_DIR="<REPO_ROOT>/detection/runs"
YOLO_OUTPUT_DIR="<REPO_ROOT>/detection/yolo/output"

find_best_sam3_ckpt() {
    local ds="$1"
    python3 << PYEOF
import re, os, glob
ds = "$ds"
runs_dir = "$DETECTION_RUNS_DIR"
run_dir = os.path.join(runs_dir, f"arrowhead_{ds}")
if not os.path.isdir(run_dir):
    print(""); exit(0)
logs = sorted(glob.glob(os.path.join(run_dir, "train_*.log")))
best_log = None
best_count = 0
for log in logs:
    with open(log) as f:
        content = f.read()
    count = content.count("Average Precision")
    if count > best_count:
        best_count = count; best_log = log
if not best_log:
    print(""); exit(0)
with open(best_log) as f:
    content = f.read()
ap_all = re.findall(r'Average Precision  \(AP\) @\[ IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = ([\d.]+)', content)
ap50  = re.findall(r'Average Precision  \(AP\) @\[ IoU=0\.50      \| area=   all \| maxDets=100 \] = ([\d.]+)', content)
ap75  = re.findall(r'Average Precision  \(AP\) @\[ IoU=0\.75      \| area=   all \| maxDets=100 \] = ([\d.]+)', content)
if not ap_all:
    print(""); exit(0)
val_epochs = list(range(5, 5 * len(ap_all) + 1, 5))
best_avg = -1; best_epoch = -1
for i in range(len(ap_all)):
    a1, a50, a75 = float(ap_all[i]), float(ap50[i]), float(ap75[i])
    avg = (a1 + a50 + a75) / 3
    if avg > best_avg:
        best_avg = avg; best_epoch = val_epochs[i]
ckpt = os.path.join(run_dir, "checkpoints", f"checkpoint_{best_epoch}.pt")
print(ckpt if os.path.exists(ckpt) else "")
PYEOF
}

find_best_yolo_ckpt() {
    local ds="$1"
    python3 << PYEOF
import csv, os, glob, re
ds = "$ds"
root = "$YOLO_OUTPUT_DIR"
run_dirs = []
single = os.path.join(root, ds)
if os.path.isdir(single):
    run_dirs.append(single)
for d in sorted(glob.glob(os.path.join(root, f"{ds}_bs*"))):
    if os.path.isdir(d):
        run_dirs.append(d)
candidates = {}
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False
for out_dir in run_dirs:
    csv_path = os.path.join(out_dir, "results.csv")
    weights_dir = os.path.join(out_dir, "weights")
    if not (os.path.isfile(csv_path) and os.path.isdir(weights_dir)):
        continue
    saved_set = set()
    for p in glob.glob(os.path.join(weights_dir, "epoch*.pt")):
        m = re.search(r"epoch(\d+)\.pt$", p)
        if m: saved_set.add(int(m.group(1)))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        map50_key = next((k for k in reader.fieldnames or [] if "mAP50(B)" in k and "mAP50-95" not in k), None)
        map5095_key = next((k for k in reader.fieldnames or [] if "mAP50-95(B)" in k), None)
        if not map50_key:
            continue
        for row in reader:
            try:
                ep = int(float(row.get("epoch", 0)))
                v50 = float(row.get(map50_key, 0) or 0)
                v5095 = float(row.get(map5095_key, 0) or 0) if map5095_key else 0.0
            except (ValueError, TypeError):
                continue
            if ep not in saved_set: continue
            ck = os.path.join(weights_dir, f"epoch{ep}.pt")
            if os.path.isfile(ck):
                candidates[ck] = (v50, v5095)
    if has_torch:
        for fname in ["best.pt", "last.pt"]:
            p = os.path.join(weights_dir, fname)
            if not os.path.isfile(p): continue
            try:
                ck_obj = torch.load(p, map_location="cpu", weights_only=False)
                tm = ck_obj.get("train_metrics", {}) or {}
                v50 = float(tm.get("metrics/mAP50(B)", 0) or 0)
                v5095 = float(tm.get("metrics/mAP50-95(B)", 0) or 0)
                candidates[p] = (v50, v5095)
            except Exception:
                pass
if not candidates:
    print(""); exit(0)
best_path = max(candidates.items(), key=lambda x: (x[1][0] + x[1][1]) / 2)[0]
print(best_path)
PYEOF
}

find_yolo_train_imgsz() {
    local ckpt_path="$1"
    python3 << PYEOF
import os, re
ckpt = "$ckpt_path"
if not ckpt:
    print(""); exit(0)
run_dir = os.path.dirname(os.path.dirname(ckpt))
args_yaml = os.path.join(run_dir, "args.yaml")
if not os.path.isfile(args_yaml):
    print(""); exit(0)
with open(args_yaml) as f:
    content = f.read()
m = re.search(r"^imgsz:\s*(\d+)", content, re.MULTILINE)
print(m.group(1) if m else "")
PYEOF
}

declare -A BEST_SAM3_CKPT
_SAM3_DATASETS=("bpmn" "flowgen_easy" "flowgen_medium" "flowgen_hard")
if [ "$SKIP_EVAL" -eq 0 ]; then
    for _ds in "${_SAM3_DATASETS[@]}"; do
        BEST_SAM3_CKPT["$_ds"]=$(find_best_sam3_ckpt "$_ds")
    done
fi

declare -A BEST_YOLO_CKPT
declare -A BEST_YOLO_IMGSZ
_YOLO_DATASETS=("bpmn" "fcb" "flowlearn" "flowvqa" "flowgen_easy" "flowgen_medium" "flowgen_hard")
if [ "$SKIP_EVAL" -eq 0 ]; then
    for _ds in "${_YOLO_DATASETS[@]}"; do
        BEST_YOLO_CKPT["$_ds"]=$(find_best_yolo_ckpt "$_ds")
        BEST_YOLO_IMGSZ["$_ds"]=$(find_yolo_train_imgsz "${BEST_YOLO_CKPT[$_ds]}")
    done
fi

declare -A YOLO_CONF_FOR
YOLO_CONF_FOR=(
    ["fcb"]=0.25 ["flowlearn"]=0.3 ["flowvqa"]=0.65 ["bpmn"]=0.3
    ["flowgen_easy"]=0.3 ["flowgen_medium"]=0.3 ["flowgen_hard"]=0.2
)
declare -A YOLO_AREA_FILTER_FOR
YOLO_AREA_FILTER_FOR=(["flowgen_hard"]=0.6)
_NO_FT_DATASETS=("fca" "cbd")

# colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================
# ① Model config
# ============================================================
declare -A MODELS
MODELS=(
    ["qwen3_vl_4b"]="<MODEL_ROOT>/Qwen3-VL-4B-Instruct"
    ["qwen3_vl_8b"]="<MODEL_ROOT>/Qwen3-VL-8B-Instruct"
)

declare -A MODEL_MAX_EPOCHS
MODEL_MAX_EPOCHS=(
    ["qwen3_vl_4b"]=10
    ["qwen3_vl_8b"]=10
)


# ============================================================
# ② Dataset config
# ============================================================
DATASETS=("flowgen_hard")

# cbd / fcb / bpmn have official val, no need from train split
declare -A HAS_OFFICIAL_VAL
HAS_OFFICIAL_VAL=(
    ["cbd"]=true
    ["fcb"]=true
    ["fca"]=false
    ["flowlearn"]=false
    ["flowvqa"]=false
    ["bpmn"]=true
    ["flowgen_easy"]=false
    ["flowgen_medium"]=false
    ["flowgen_hard"]=false
)

# ============================================================
# ③ Batch Size grid search space
# ============================================================
declare -A BS_GRID
BS_GRID=(
    ["fca"]="8 16 32 64 128"
    ["cbd"]="8 16 32 64 128"
    ["fcb"]="8 16 32 64 128"
    ["flowlearn"]="8 16 32 64 128"
    ["flowvqa"]="8 16 32 64 128"
    ["bpmn"]="8 16 32 64 128"
    ["flowgen_easy"]="8 16 32 64 128"
    ["flowgen_medium"]="8 16 32 64 128"
    ["flowgen_hard"]="8 16 32 64 128"
)

# ============================================================
# ④ early stopping params
# ============================================================
EARLY_STOPPING_PATIENCE=1
EARLY_STOPPING_THRESHOLD=0.0
VAL_RATIO=0.1
VAL_SPLIT_SEED=42

# ============================================================
# ⑤ Path config
# ============================================================
ARROW_DATA_DIR="<DATA_ROOT>/data_4_training"
OUTPUT_BASE_DIR="${SCRIPT_CWD}/output"
SPLIT_CACHE_DIR="${SCRIPT_CWD}/output/_arrow_train_val_splits"
EVAL_OUTPUT_DIR="<OUTPUT_ROOT>/output_lora_arrow"
TRAIN_SCRIPT="${SCRIPT_CWD}/scripts/train_early_stop.sh"
SPLIT_SCRIPT="${SCRIPT_CWD}/scripts/split_train_val.py"

# cbd/fcb/bpmn official val arrow data path (by gen_val_arrow_data.py generate)
declare -A OFFICIAL_VAL_DATA
OFFICIAL_VAL_DATA=(
    ["cbd"]="${ARROW_DATA_DIR}/cbd_val_arrow_training_all.json"
    ["fcb"]="${ARROW_DATA_DIR}/fcb_val_arrow_training_all.json"
    ["bpmn"]="${ARROW_DATA_DIR}/bpmn_val_arrow_training_all.json"
)

# ============================================================
# ⑤-b each dataset test set config (for Stage 3 run test)
# ============================================================
declare -A TEST_IMAGE_DIR
TEST_IMAGE_DIR=(
    ["fca"]="<DATA_ROOT>/Block-Diagram-Datasets/Handwritten_block_diagrams/FC_A/test_img"
    ["cbd"]="<DATA_ROOT>/Block-Diagram-Datasets/Computerized_block_diagrams(CBD)/test_img"
    ["fcb"]="<DATA_ROOT>/Block-Diagram-Datasets/Handwritten_block_diagrams/FC_B/test"
    ["flowlearn"]="<DATA_ROOT>/flowlearn/mermaid_word/jpeg"
    ["flowvqa"]="<DATA_ROOT>/flowvqa/Data/A. Main Set Flowchart Images"
    ["bpmn"]="<DATA_ROOT>/bpmn/test_images"
    # FlowGen: the official test images (used by sam / sam3_ft / yolo)
    ["flowgen_easy"]="<DATA_ROOT>/flowgen/test_img_easy_orig"
    ["flowgen_medium"]="<DATA_ROOT>/flowgen/test_img_medium_orig"
    ["flowgen_hard"]="<DATA_ROOT>/flowgen/test_img_hard_orig"
)

declare -A TEST_JSON
TEST_JSON=(
    ["fca"]=""
    ["cbd"]=""
    ["fcb"]=""
    ["flowlearn"]="<DATA_ROOT>/flowlearn/test.json"
    ["flowvqa"]="<DATA_ROOT>/flowvqa/Data/test_full.json"
    ["bpmn"]="<DATA_ROOT>/bpmn/test.json"
)

EVAL_PY_BIN="<REPO_ROOT>/lora/eval/eval_TRACE.py"

# dataset name → eval_TRACE.py --dataset value (flowgen_{easy,medium,hard} all map to flowgen)
declare -A EVAL_DATASET
EVAL_DATASET=(
    ["fca"]="fca"
    ["cbd"]="cbd"
    ["fcb"]="fcb"
    ["flowlearn"]="flowlearn"
    ["flowvqa"]="flowvqa"
    ["bpmn"]="bpmn"
    ["flowgen_easy"]="flowgen"
    ["flowgen_medium"]="flowgen"
    ["flowgen_hard"]="flowgen"
)

# flowgen evaluation need --gt-root point to matching difficulty GT dir
declare -A EVAL_GT_ROOT
EVAL_GT_ROOT=(
    ["flowgen_easy"]="<DATA_ROOT>/data_4_training/flowgen_test_easy"
    ["flowgen_medium"]="<DATA_ROOT>/data_4_training/flowgen_test_medium"
    ["flowgen_hard"]="<DATA_ROOT>/data_4_training/flowgen_test_hard"
)

# FlowGen note: the official FlowGen test images have rendering glitches on a few diagrams.
# sam / sam3_ft / yolo run on the official images (TEST_IMAGE_DIR above) for a fair, like-for-like
# comparison. The groundtruth arrow boxes, however, were labelled on OUR re-rendered images, so for
# arrow-source=groundtruth we feed those re-rendered images instead — otherwise the GT bbox
# coordinates would not line up with the official images. Only the groundtruth branch uses this map.
declare -A FLOWGEN_GT_IMAGE_DIR
FLOWGEN_GT_IMAGE_DIR=(
    ["flowgen_easy"]="<DATA_ROOT>/flowgen/test_img_easy"
    ["flowgen_medium"]="<DATA_ROOT>/flowgen/test_img_medium"
    ["flowgen_hard"]="<DATA_ROOT>/flowgen/test_img_hard"
)

# ============================================================
# ⑥ GPU config
# ============================================================
CUDA_DEVICES="0,1,2,3,4,5,6,7"
NUM_DEVICES=8
INFERENCE_GPUS="0,1,2,3,4,5,6,7"

# ============================================================
# ⑦ logging
# ============================================================
LOG_DIR="${SCRIPT_CWD}/training_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/arrow_grid_search_${TIMESTAMP}.log"

log() { echo -e "$@" | tee -a "$MAIN_LOG"; }

log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  Arrow Grid Search training script (Batch Size + Early Stopping)${NC}"
log "${BLUE}  Start Time: $(date)${NC}"
log "${BLUE}  CUDA: ${CUDA_DEVICES}  |  Devices: ${NUM_DEVICES}${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""
log "${CYAN}[training paradigm]${NC}"
log "  1. cbd/fcb: use official val arrow data (not split train) "
log "  2. other dataset: from train split val (val_ratio=${VAL_RATIO})"
log "  3. Batch Size grid search"
log "  4. early stopping: patience=${EARLY_STOPPING_PATIENCE}"
log "  5. save/eval strategy: epoch"
log "  6. across BS compare best val loss → pick best → run directly test → produce final score"
log ""

# ============================================================
# ⑧ Stage 1: prepare val data
# ============================================================
log "${CYAN}[Stage 1] prepare val data${NC}"
log ""

mkdir -p "$SPLIT_CACHE_DIR"

for dataset in "${DATASETS[@]}"; do
    # flowgen_* dataset training file naming differs
    if [[ "$dataset" == flowgen_* ]]; then
        ORIG_DATA="${ARROW_DATA_DIR}/${dataset/flowgen_/flowgen_train_}.json"
    else
        ORIG_DATA="${ARROW_DATA_DIR}/${dataset}_arrow_training_all.json"
    fi

    if [ ! -f "$ORIG_DATA" ]; then
        log "${RED}  ✗ ${dataset}: training data not found: ${ORIG_DATA}${NC}"
        continue
    fi

    if [ "${HAS_OFFICIAL_VAL[$dataset]}" = "true" ]; then
        # has official val: directly use, train keep complete
        VAL_FILE="${OFFICIAL_VAL_DATA[$dataset]}"
        if [ ! -f "$VAL_FILE" ]; then
            log "${RED}  ✗ ${dataset}: official val data not found: ${VAL_FILE}${NC}"
            log "    first run: cd "<REPO_ROOT>" && python -m gen_data.prepare_training_data --datasets ${dataset}"
            continue
        fi
        # training use full data, val use official data
        ln -sf "$ORIG_DATA" "${SPLIT_CACHE_DIR}/${dataset}_train.json"
        ln -sf "$VAL_FILE" "${SPLIT_CACHE_DIR}/${dataset}_val.json"
        N_TRAIN=$(python3 -c "import json; print(len(json.load(open('$ORIG_DATA'))))")
        N_VAL=$(python3 -c "import json; print(len(json.load(open('$VAL_FILE'))))")
        log "  ${dataset}: official val (train=${N_TRAIN}, val=${N_VAL})"
    else
        # no official val: from train split
        TRAIN_SPLIT="${SPLIT_CACHE_DIR}/${dataset}_train.json"
        VAL_SPLIT="${SPLIT_CACHE_DIR}/${dataset}_val.json"

        if [ -f "$TRAIN_SPLIT" ] && [ -f "$VAL_SPLIT" ] && [ "$TRAIN_SPLIT" -nt "$ORIG_DATA" ]; then
            N_TRAIN=$(python3 -c "import json; print(len(json.load(open('$TRAIN_SPLIT'))))")
            N_VAL=$(python3 -c "import json; print(len(json.load(open('$VAL_SPLIT'))))")
            log "  ${dataset}: reuse existing split (train=${N_TRAIN}, val=${N_VAL})"
        else
            log "  ${dataset}: from train split (val_ratio=${VAL_RATIO})..."
            python3 "$SPLIT_SCRIPT" \
                --input "$ORIG_DATA" \
                --train-output "$TRAIN_SPLIT" \
                --val-output "$VAL_SPLIT" \
                --val-ratio "$VAL_RATIO" \
                --seed "$VAL_SPLIT_SEED" \
                2>&1 | while read line; do log "    $line"; done
        fi
    fi
done

# ── Shuffle all dataset training data (seed fixed, ensure reproducibility) ──
for dataset in "${DATASETS[@]}"; do
    TRAIN_FILE="${SPLIT_CACHE_DIR}/${dataset}_train.json"
    if [ -L "$TRAIN_FILE" ]; then
        REAL_FILE=$(readlink -f "$TRAIN_FILE")
    else
        REAL_FILE="$TRAIN_FILE"
    fi
    if [ -f "$REAL_FILE" ]; then
        log "  ${dataset}: Shuffling training data (seed=${VAL_SPLIT_SEED})..."
        python3 -c "
import json, random
random.seed(${VAL_SPLIT_SEED})
with open('${REAL_FILE}') as f:
    data = json.load(f)
random.shuffle(data)
with open('${REAL_FILE}', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f'  Shuffled {len(data)} records → ${REAL_FILE}')
" 2>&1 | while read line; do log "  $line"; done
    fi
done

log ""

# ============================================================
# ⑨ Stage 2: Grid Search training
# ============================================================
log "${CYAN}[Stage 2] Grid Search training (Arrow)${NC}"
log ""

TOTAL_TASKS=0
for model_key in "${!MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for bs in ${BS_GRID[$dataset]}; do
            TOTAL_TASKS=$((TOTAL_TASKS + 1))
        done
    done
done
COMPLETED_TASKS=0
FAILED_TASKS=0
SKIPPED_TASKS=0

log "total training tasks: ${TOTAL_TASKS}"
log ""

# search space preview
log "${CYAN}[search space preview]${NC}"
log "$(printf '  %-16s %-12s %-30s %-10s %-10s' 'Model' 'Dataset' 'BS Grid' 'MaxEpochs' 'ValSource')"
log "$(printf '  %-16s %-12s %-30s %-10s %-10s' '-----' '-------' '-------' '---------' '---------')"
for model_key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    MAX_EP=${MODEL_MAX_EPOCHS[$model_key]}
    for dataset in "${DATASETS[@]}"; do
        val_src="split"
        [ "${HAS_OFFICIAL_VAL[$dataset]}" = "true" ] && val_src="official"
        log "$(printf '  %-16s %-12s %-30s %-10s %-10s' "$model_key" "$dataset" "${BS_GRID[$dataset]}" "$MAX_EP" "$val_src")"
    done
done
log ""

# main training loop
for model_key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    MODEL_PATH="${MODELS[$model_key]}"
    MAX_EPOCHS="${MODEL_MAX_EPOCHS[$model_key]}"

    log "${BLUE}════════════════════════════════════════════════════════════${NC}"
    log "${BLUE}  model: ${model_key}  (max_epochs=${MAX_EPOCHS})${NC}"
    log "${BLUE}════════════════════════════════════════════════════════════${NC}"

    for dataset in "${DATASETS[@]}"; do
        TRAIN_DATA="${SPLIT_CACHE_DIR}/${dataset}_train.json"
        VAL_DATA="${SPLIT_CACHE_DIR}/${dataset}_val.json"

        # handle symlink
        if [ -L "$TRAIN_DATA" ]; then
            TRAIN_DATA=$(readlink -f "$TRAIN_DATA")
        fi
        if [ -L "$VAL_DATA" ]; then
            VAL_DATA=$(readlink -f "$VAL_DATA")
        fi

        if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
            log "${RED}  ✗ ${dataset}: training/validation data missing, skip${NC}"
            for bs in ${BS_GRID[$dataset]}; do
                SKIPPED_TASKS=$((SKIPPED_TASKS + 1))
            done
            continue
        fi

        N_TRAIN=$(python3 -c "import json; print(len(json.load(open('$TRAIN_DATA'))))")
        N_VAL=$(python3 -c "import json; print(len(json.load(open('$VAL_DATA'))))")

        log ""
        log "${YELLOW}  ── ${dataset} (train=${N_TRAIN}, val=${N_VAL}) ──${NC}"

        for bs in ${BS_GRID[$dataset]}; do
            # Note: arrow version uses _arrow_ distinguished from triplet version
            RUN_NAME="${model_key}_arrow_${dataset}_bs${bs}"
            RUN_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_NAME}"
            TASK_LOG="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"

            log ""
            log "${GREEN}  ▶ [${COMPLETED_TASKS}/${TOTAL_TASKS}] ${model_key} × ${dataset} × bs=${bs}${NC}"

            # check whether a checkpoint already exists: resume training (resume) rather than skip
            if [ -d "$RUN_OUTPUT_DIR" ]; then
                CKPT_COUNT=$(find "$RUN_OUTPUT_DIR" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)
                if [ "$CKPT_COUNT" -gt 0 ]; then
                    log "    ${CYAN}already have ${CKPT_COUNT}   checkpoint, willresume from latest checkpoint (resume)${NC}"
                fi
            fi

            log "    epochs=${MAX_EPOCHS}  bs=${bs}  patience=${EARLY_STOPPING_PATIENCE}"
            log "    train: ${TRAIN_DATA}"
            log "    val:   ${VAL_DATA}"
            log "    out:   ${RUN_OUTPUT_DIR}"

            bash "$TRAIN_SCRIPT" \
                --model_name "$MODEL_PATH" \
                --cuda_devices "$CUDA_DEVICES" \
                --data_path "$TRAIN_DATA" \
                --eval_data_path "$VAL_DATA" \
                --output_dir "$RUN_OUTPUT_DIR" \
                --num_epochs "$MAX_EPOCHS" \
                --batch_size "$bs" \
                --num_devices "$NUM_DEVICES" \
                --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
                --early_stopping_threshold "$EARLY_STOPPING_THRESHOLD" \

                > "$TASK_LOG" 2>&1

            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                FINAL_CKPTS=$(find "$RUN_OUTPUT_DIR" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)
                STOPPED_EPOCH=$(python3 -c "
import json, glob, os
# supports both layouts: top level or checkpoint subdir
state_file = '${RUN_OUTPUT_DIR}/trainer_state.json'
if os.path.isfile(state_file):
    with open(state_file) as f:
        state = json.load(f)
    print(state.get('epoch', '?'))
else:
    state_files = glob.glob('${RUN_OUTPUT_DIR}/checkpoint-*/trainer_state.json')
    if state_files:
        state_files.sort(key=lambda x: int(x.split('checkpoint-')[1].split('/')[0]))
        with open(state_files[-1]) as f:
            state = json.load(f)
        print(state.get('epoch', '?'))
    else:
        print('?')
" 2>/dev/null)
                log "    ${GREEN}✓ success (stopped at epoch ${STOPPED_EPOCH}, ${FINAL_CKPTS}   checkpoint)${NC}"
                COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
            else
                log "    ${RED}✗ failed (exit=${EXIT_CODE}), see: ${TASK_LOG}${NC}"
                tail -5 "$TASK_LOG" 2>/dev/null | while read line; do log "      $line"; done
                FAILED_TASKS=$((FAILED_TASKS + 1))
            fi

            log "    ${BLUE}progress: done=${COMPLETED_TASKS} skip=${SKIPPED_TASKS} failed=${FAILED_TASKS} / total=${TOTAL_TASKS}${NC}"
        done
    done
done

# ============================================================
# ⑩ Stage 3: across BS pick best → test inference (multiple arrow source) → scoring
# ============================================================
if [ "$SKIP_EVAL" -eq 1 ]; then
    log ""
    log "${YELLOW}[Stage 3] skip (--skip-eval)${NC}"
else
    log ""
    log "${CYAN}[Stage 3] across Batch Size pick best checkpoint → test inference (${ARROW_SOURCES}) → scoring${NC}"
    log ""

    log "${CYAN}SAM3 detector (by dataset auto-select) :${NC}"
    for _ds in "${_SAM3_DATASETS[@]}"; do
        if [ -n "${BEST_SAM3_CKPT[$_ds]}" ]; then
            log "  ${_ds}: ${BEST_SAM3_CKPT[$_ds]}"
        else
            log "  ${_ds}: (no fine-tuned detector)"
        fi
    done
    log ""
    log "${CYAN}YOLO detector (by dataset auto-select) :${NC}"
    for _ds in "${_YOLO_DATASETS[@]}"; do
        if [ -n "${BEST_YOLO_CKPT[$_ds]}" ]; then
            log "  ${_ds}: ${BEST_YOLO_CKPT[$_ds]}  (imgsz=${BEST_YOLO_IMGSZ[$_ds]:-?})"
        else
            log "  ${_ds}: (no YOLO detector)"
        fi
    done
    log ""

    MODELS_KEYS_STR=$(echo "${!MODELS[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')
    BS_GRIDS_STR=""
    for dataset in "${DATASETS[@]}"; do
        bs_list=$(echo "${BS_GRID[$dataset]}" | tr ' ' ',')
        BS_GRIDS_STR="${BS_GRIDS_STR} ${dataset}:${bs_list}"
    done

    # ── selecteach (model, dataset) best BS + checkpoint (by val_loss) ──
    BEST_CONFIGS=$(python3 - "$OUTPUT_BASE_DIR" "$MODELS_KEYS_STR" "${DATASETS[*]}" "$BS_GRIDS_STR" "arrow" << 'PYEOF'
import json, glob, os, sys
output_base, models_str, datasets_str, bs_grids_str, task_type = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
models = models_str.split()
datasets = datasets_str.split()
bs_grids = {}
for item in bs_grids_str.split():
    ds, bss = item.split(":")
    bs_grids[ds] = bss.split(",")

for model_key in models:
    for dataset in datasets:
        best_val_loss = float("inf")
        best_info = None
        for bs in bs_grids.get(dataset, []):
            run_dir = os.path.join(output_base, f"{model_key}_{task_type}_{dataset}_bs{bs}")
            if not os.path.isdir(run_dir):
                continue
            all_evals = []
            ckpt_dirs = sorted(
                glob.glob(os.path.join(run_dir, "checkpoint-*")),
                key=lambda x: int(x.split("checkpoint-")[1])
            )
            for cd in ckpt_dirs:
                sf = os.path.join(cd, "trainer_state.json")
                if not os.path.isfile(sf):
                    continue
                with open(sf) as f:
                    st = json.load(f)
                evals = [e for e in st.get("log_history", []) if "eval_loss" in e]
                if evals:
                    last_e = evals[-1]
                    all_evals.append((last_e.get("step", 0), last_e["eval_loss"], cd, last_e.get("epoch", "?")))
            if not all_evals:
                top = os.path.join(run_dir, "trainer_state.json")
                if os.path.isfile(top):
                    with open(top) as f:
                        st = json.load(f)
                    for e in [x for x in st.get("log_history", []) if "eval_loss" in x]:
                        step = e.get("step", 0)
                        cp = os.path.join(run_dir, f"checkpoint-{step}")
                        if not os.path.isdir(cp): cp = run_dir
                        all_evals.append((step, e["eval_loss"], cp, e.get("epoch", "?")))
            if not all_evals:
                continue
            be = min(all_evals, key=lambda x: x[1])
            if be[1] < best_val_loss:
                best_val_loss = be[1]
                best_info = (bs, be[1], be[2], be[3], be[0])
        if best_info:
            bs, vl, cd, ep, st = best_info
            print(f"{model_key}|{dataset}|{bs}|{vl:.6f}|{cd}|{ep}|{st}")
PYEOF
)

    if [ -z "$BEST_CONFIGS" ]; then
        log "${RED}no valid training results found, skip test eval${NC}"
    else
        log "best configs found:"
        log "$(printf '  %-16s %-12s %-6s %-12s %-8s' 'Model' 'Dataset' 'BS' 'ValLoss' 'Epoch')"
        while IFS='|' read -r model_key dataset bs val_loss ckpt_dir epoch step; do
            log "$(printf '  %-16s %-12s %-6s %-12s %-8s' "$model_key" "$dataset" "$bs" "$val_loss" "$epoch")"
        done <<< "$BEST_CONFIGS"
        log ""

        mkdir -p "$EVAL_OUTPUT_DIR"
        declare -A FINAL_RESULTS

        TASK_COUNT=0
        COMPLETED_COUNT=0
        FAILED_COUNT=0
        for _as in $ARROW_SOURCES; do
            while IFS='|' read -r _ _ _ _ _ _ _; do
                TASK_COUNT=$((TASK_COUNT + 1))
            done <<< "$BEST_CONFIGS"
        done

        for ARROW_SOURCE_CUR in $ARROW_SOURCES; do
            log ""
            log "${BLUE}──── Arrow Source: ${ARROW_SOURCE_CUR} ────${NC}"

            while IFS='|' read -r model_key dataset best_bs best_val_loss best_ckpt_dir stopped_epoch best_step; do
                MODEL_PATH="${MODELS[$model_key]}"
                T_JSON="${TEST_JSON[$dataset]}"
                EVAL_PY="$EVAL_PY_BIN"
                EVAL_DS="${EVAL_DATASET[$dataset]}"

                IMAGE_DIR="${TEST_IMAGE_DIR[$dataset]}"
                # FlowGen groundtruth only: the GT arrow boxes were labelled on our re-rendered
                # images, so swap them in (the official images' coords would not match). Every other
                # arrow source keeps the official images. See FLOWGEN_GT_IMAGE_DIR above.
                if [[ "$dataset" == flowgen_* && "$ARROW_SOURCE_CUR" == "groundtruth" ]]; then
                    IMAGE_DIR="${FLOWGEN_GT_IMAGE_DIR[$dataset]:-$IMAGE_DIR}"
                fi

                if [ -z "$MODEL_PATH" ] || [ -z "$IMAGE_DIR" ] || [ -z "$EVAL_PY" ]; then
                    log "${YELLOW}  ⊘ ${model_key}/${dataset}/${ARROW_SOURCE_CUR}: incomplete config, skip${NC}"
                    continue
                fi

                DS_ARROW_SOURCE="$ARROW_SOURCE_CUR"
                DS_SAM3_CKPT="${BEST_SAM3_CKPT[$dataset]}"
                DS_YOLO_CKPT="${BEST_YOLO_CKPT[$dataset]}"
                DS_YOLO_IMGSZ="${BEST_YOLO_IMGSZ[$dataset]}"

                # fca/cbd no SAM3/YOLO fine-tuned detector, fall back to sam
                for _nft in "${_NO_FT_DATASETS[@]}"; do
                    if [[ "$dataset" == "$_nft" ]]; then
                        if [[ "$DS_ARROW_SOURCE" == "sam3_ft" || "$DS_ARROW_SOURCE" == "yolo" ]]; then
                            log "    ${YELLOW}⚠ ${dataset} no fine-tuned detector, ${DS_ARROW_SOURCE} fall back to sam${NC}"
                            DS_ARROW_SOURCE="sam"
                        fi
                        break
                    fi
                done
                if [[ "$DS_ARROW_SOURCE" == "sam3_ft" && -z "$DS_SAM3_CKPT" ]]; then
                    log "    ${YELLOW}⚠ ${dataset} no SAM3 detector, fall back to sam${NC}"
                    DS_ARROW_SOURCE="sam"
                fi
                if [[ "$DS_ARROW_SOURCE" == "yolo" && -z "$DS_YOLO_CKPT" ]]; then
                    log "    ${YELLOW}⚠ ${dataset} no YOLO detector, fall back to sam${NC}"
                    DS_ARROW_SOURCE="sam"
                fi

                RESULT_DIR="${EVAL_OUTPUT_DIR}/${model_key}_arrow_${dataset}_bs${best_bs}_best_${DS_ARROW_SOURCE}"
                mkdir -p "$RESULT_DIR"

                log "${GREEN}  ▶ [${COMPLETED_COUNT}/${TASK_COUNT}] ${model_key}/${dataset}/${DS_ARROW_SOURCE}${NC}"
                log "    BS=${best_bs}  val_loss=${best_val_loss}  ckpt=${best_ckpt_dir}"

                # -- inference --
                INFER_CMD="python -m lora.inference.get_triplets_sft_trace"
                INFER_CMD="${INFER_CMD} --output-dir \"${RESULT_DIR}\""
                INFER_CMD="${INFER_CMD} --image-dir \"${IMAGE_DIR}\""
                INFER_CMD="${INFER_CMD} --backend local"
                INFER_CMD="${INFER_CMD} --adapter-path ${best_ckpt_dir}"
                INFER_CMD="${INFER_CMD} --base-model-path ${MODEL_PATH}"
                INFER_CMD="${INFER_CMD} --gpus ${INFERENCE_GPUS}"
                INFER_CMD="${INFER_CMD} --workers-per-gpu 2"
                INFER_CMD="${INFER_CMD} --arrow-source ${DS_ARROW_SOURCE}"
                if [[ "$DS_ARROW_SOURCE" == "sam3_ft" ]]; then
                    INFER_CMD="${INFER_CMD} --sam3-ft-checkpoint ${DS_SAM3_CKPT}"
                elif [[ "$DS_ARROW_SOURCE" == "yolo" ]]; then
                    INFER_CMD="${INFER_CMD} --yolo-checkpoint ${DS_YOLO_CKPT}"
                    [ -n "$DS_YOLO_IMGSZ" ] && INFER_CMD="${INFER_CMD} --yolo-imgsz ${DS_YOLO_IMGSZ}"
                    [ -n "${YOLO_CONF_FOR[$dataset]}" ] && INFER_CMD="${INFER_CMD} --yolo-conf ${YOLO_CONF_FOR[$dataset]}"
                    [ -n "${YOLO_AREA_FILTER_FOR[$dataset]}" ] && INFER_CMD="${INFER_CMD} --yolo-area-filter-median-ratio ${YOLO_AREA_FILTER_FOR[$dataset]}"
                fi
                [ -n "$T_JSON" ] && INFER_CMD="${INFER_CMD} --test-json ${T_JSON}"
                if [[ "$dataset" == flowgen_* ]]; then
                    INFER_CMD="${INFER_CMD} --is-flowgen"
                elif [[ "$dataset" == "bpmn" ]]; then
                    INFER_CMD="${INFER_CMD} --is-bpmn"
                fi

                INFER_LOG="${LOG_DIR}/infer_${model_key}_${dataset}_bs${best_bs}_${DS_ARROW_SOURCE}_${TIMESTAMP}.log"

                # resume: use test_json entry count or dir image count asastotalsample count
                EXISTING_RESULTS=$(find "$RESULT_DIR" -name "arrow_triplets.json" -type f 2>/dev/null | wc -l)
                if [ -n "$T_JSON" ] && [ -f "$T_JSON" ]; then
                    TOTAL_IMAGES=$(python3 -c "
import json
with open('$T_JSON') as f:
    d = json.load(f)
print(len(d) if isinstance(d, (list, dict)) else 0)
" 2>/dev/null)
                    [ -z "$TOTAL_IMAGES" ] && TOTAL_IMAGES=0
                else
                    TOTAL_IMAGES=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
                fi
                if [ "$EXISTING_RESULTS" -ge "$TOTAL_IMAGES" ] && [ "$TOTAL_IMAGES" -gt 0 ]; then
                    log "    inference already done (${EXISTING_RESULTS}/${TOTAL_IMAGES}), skip"
                else
                    log "    inferring... (already have ${EXISTING_RESULTS}/${TOTAL_IMAGES})"
                    (cd "<REPO_ROOT>" && eval ${INFER_CMD}) > "${INFER_LOG}" 2>&1
                    if [ $? -ne 0 ]; then
                        log "    ${RED}✗ inference failed, see ${INFER_LOG}${NC}"
                        tail -10 "$INFER_LOG" 2>/dev/null | while read line; do log "      $line"; done
                        FAILED_COUNT=$((FAILED_COUNT + 1))
                        continue
                    fi
                fi

                # -- scoring --
                if [ ! -f "$EVAL_PY" ]; then
                    log "    ${RED}✗ eval script not found: ${EVAL_PY}${NC}"
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                    continue
                fi

                EVAL_EXTRA_ARGS=""
                [ -n "${EVAL_GT_ROOT[$dataset]}" ] && EVAL_EXTRA_ARGS="--gt-root ${EVAL_GT_ROOT[$dataset]}"
                EVAL_OUTPUT=$(cd "<REPO_ROOT>" && python "$EVAL_PY" --dataset "$EVAL_DS" --output-dir "$RESULT_DIR" $EVAL_EXTRA_ARGS --image-dir "$IMAGE_DIR" --test-json "${T_JSON:-}" 2>&1)
                if [ $? -ne 0 ]; then
                    log "    ${RED}✗ scoring failed${NC}"
                    log "    ${EVAL_OUTPUT}"
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                    continue
                fi

                SCORE=$(python3 -c "
import json, os
p = os.path.join('${RESULT_DIR}', 'evaluation_results.json')
if os.path.isfile(p):
    d = json.load(open(p))
    print(f'{d.get(\"f1_score\", d.get(\"relaxed_f1\", 0)):.4f}')
else:
    print('N/A')
" 2>/dev/null)

                log "    ${GREEN}✓ F1 = ${SCORE}${NC}"
                FINAL_RESULTS["${model_key}|${dataset}|${DS_ARROW_SOURCE}"]="${best_bs}|${best_val_loss}|${SCORE}|${stopped_epoch}|${best_step}"
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))

            done <<< "$BEST_CONFIGS"
        done

        # ── summarytable ──
        log ""
        log "${BLUE}════════ Arrow evaluation final result ════════${NC}"
        log "$(printf '  %-16s %-14s %-12s %-6s %-12s %-12s %-8s' 'Model' 'Dataset' 'ArrowSrc' 'BS' 'ValLoss' 'TestF1' 'Epoch')"
        for key in $(echo "${!FINAL_RESULTS[@]}" | tr ' ' '\n' | sort); do
            IFS='|' read -r mk ds asrc <<< "$key"
            IFS='|' read -r bs vl sc ep st <<< "${FINAL_RESULTS[$key]}"
            log "$(printf '  %-16s %-14s %-12s %-6s %-12s %-12s %-8s' "$mk" "$ds" "$asrc" "$bs" "$vl" "$sc" "$ep")"
        done

        SUMMARY_JSON="${EVAL_OUTPUT_DIR}/arrow_eval_results_${TIMESTAMP}.json"
        python3 -c "
import json
results = {}
raw = '''$(for key in "${!FINAL_RESULTS[@]}"; do echo "${key}=${FINAL_RESULTS[$key]}"; done)'''
for line in raw.strip().split('\n'):
    if not line.strip(): continue
    k, v = line.split('=', 1)
    model, dataset, arrow_src = k.split('|')
    bs, vl, sc, ep, st = v.split('|')
    results[f'{model}/{dataset}/{arrow_src}'] = {
        'best_batch_size': int(bs), 'best_val_loss': float(vl),
        'test_f1_score': sc, 'stopped_epoch': ep, 'best_step': st,
    }
with open('$SUMMARY_JSON', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'results saved: $SUMMARY_JSON')
" 2>&1 | while read line; do log "  $line"; done
    fi
fi

# ============================================================
# ⑪ summary
# ============================================================
log ""
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  Arrow Grid Search all done!  End Time: $(date)${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "  training tasks: ${TOTAL_TASKS}  done: ${COMPLETED_TASKS}  skip: ${SKIPPED_TASKS}  failed: ${FAILED_TASKS}"
log "  main log:   ${MAIN_LOG}"
log "  training output: ${OUTPUT_BASE_DIR}/"
log "  eval output: ${EVAL_OUTPUT_DIR}/"

if [ "$FAILED_TASKS" -gt 0 ]; then
    log "${RED}⚠ has ${FAILED_TASKS}  training tasksfailed${NC}"
    exit 1
fi
log "${GREEN}✅ all done${NC}"
exit 0
