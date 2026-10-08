#!/bin/bash

# ============================================================
# train_all_grid_search.sh
#
# newtraining paradigm:
# 1. auto-split training data into train(90%) / val(10%)
# 2. for each (model, dataset) combo, in Batch Size search over grid
# 3. each config use early stoppingmechanism (patience=1), each epoch evaluation once
# 4. training done after, across BS compare best val loss, pick best config
# 5. use that best checkpoint run directly test inference + scoring, get final score
#
# Usage:
# cd MLLMs_SFT/Qwen-VL-Series-Finetune
# bash scripts/train_all_grid_search.sh
# ============================================================

# all paths relative to script launch working dir

# ============================================================
# *** LLaVA-v1.6-Mistral-7B-hf (ms-swift) ***
# - model path: "<MODEL_ROOT>/llava-v1.6-mistral-7b-hf"
# - TRAIN_SCRIPT switched to the swift version in this dir train_early_stop.sh.
# - Stage 3 (test inference + scoring) use lora.inference.get_triplets_swift_e2e (generic HF VLM version provided)
# this script uses AutoModelForImageTextToText + PEFT to load base + LoRA,
# output single prediction JSON, scored with lora/eval/eval_E2E.py --prediction-file.
# runs end-to-end once the model path is in place; to short-circuit set SKIP_INFER=1.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_CWD="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SCRIPT_CWD"

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
    ["llava_v16"]="<MODEL_ROOT>/llava-v1.6-mistral-7b-hf"
)

declare -A MODEL_MAX_EPOCHS
MODEL_MAX_EPOCHS=(
    ["llava_v16"]=10
)


# ============================================================
# ② Dataset config
# ============================================================
DATASETS=("flowgen_hard" "flowvqa")

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
TRIPLET_DATA_DIR="<DATA_ROOT>/data_4_triplet_training"
OUTPUT_BASE_DIR="${SCRIPT_CWD}/output"
SPLIT_CACHE_DIR="${SCRIPT_CWD}/output/_train_val_splits"
EVAL_OUTPUT_DIR="<OUTPUT_ROOT>/output_lora_triplet"
TRAIN_SCRIPT="${SCRIPT_CWD}/scripts/train_early_stop.sh"
SPLIT_SCRIPT="${SCRIPT_CWD}/scripts/split_train_val.py"

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
    ["flowgen_easy"]="<DATA_ROOT>/flowgen/test_img_easy_orig"
    ["flowgen_medium"]="<DATA_ROOT>/flowgen/test_img_medium_orig"
    ["flowgen_hard"]="<DATA_ROOT>/flowgen/test_img_hard_orig"
)

declare -A TEST_JSON
TEST_JSON=(
    ["fca"]=""
    ["cbd"]=""
    ["bpmn"]=""
    ["fcb"]=""
    ["flowlearn"]="<DATA_ROOT>/flowlearn/test.json"
    ["flowvqa"]="<DATA_ROOT>/flowvqa/Data/test_full.json"
    ["flowgen_easy"]=""
    ["flowgen_medium"]=""
    ["flowgen_hard"]=""
)

declare -A TEST_IMAGE_EXT
TEST_IMAGE_EXT=(
    ["fca"]=".png"
    ["cbd"]=".png"
    ["fcb"]=".png"
    ["flowlearn"]=".jpeg"
    ["flowvqa"]=".png"
    ["bpmn"]=".png"
    ["flowgen_easy"]=".png"
    ["flowgen_medium"]=".png"
    ["flowgen_hard"]=".png"
)

declare -A TEST_MAX_TOKENS
TEST_MAX_TOKENS=(
    ["fca"]="1024"
    ["cbd"]="2048"
    ["fcb"]="2048"
    ["flowlearn"]="2048"
    ["flowvqa"]="4096"
    ["bpmn"]="8192"
    ["flowgen_easy"]="4096"
    ["flowgen_medium"]="8192"
    ["flowgen_hard"]="8192"
)

EVAL_PY_BIN="<REPO_ROOT>/lora/eval/eval_E2E.py"
declare -A EVAL_DATASET
EVAL_DATASET=(
    ["fca"]="fca" ["cbd"]="cbd" ["fcb"]="fcb"
    ["flowlearn"]="flowlearn" ["flowvqa"]="flowvqa" ["bpmn"]="bpmn"
    ["flowgen_easy"]="flowgen" ["flowgen_medium"]="flowgen" ["flowgen_hard"]="flowgen"
)

# ============================================================
# ⑤-c has official val dataset → directly use official val, no random split
# train: use full ${dataset}_triplet_training_all.json (no truncation)
# val: use specified below official val file
# ============================================================
declare -A OFFICIAL_VAL_DATA
OFFICIAL_VAL_DATA=(
    ["cbd"]="${TRIPLET_DATA_DIR}/cbd_val_triplet_training_all.json"
    ["fcb"]="${TRIPLET_DATA_DIR}/fcb_val_triplet_training_all.json"
    ["bpmn"]="${TRIPLET_DATA_DIR}/bpmn_triplet_dev.json"
)

# ============================================================
# ⑥ GPU config
# ============================================================

# ⚠ Multi-GPU warning:
# swift 4.1.2 + torchrun + bitsandbytes has CUDA device-probing conflicts under multi-GPU,
# shows up in crash logs as "device >= 0 && device < num_gpus INTERNAL ASSERT FAILED ...
# device=N, num_gpus=N". this usually happens when a card has zombie processes left from a previous failed run.
# first use `nvidia-smi` confirm no stale processes of your own on the card; to clear use `pkill -u $USER -f swift`
# or similar. to restore multi-GPU set NUM_DEVICES back to 6 (or whatever number you want), and update CUDA_DEVICES.
CUDA_DEVICES="0,1,2,3,4,5,6,7"
NUM_DEVICES=8
INFERENCE_GPUS="0,1,2,3,4,5,6,7"

# ============================================================
# ⑦ logging
# ============================================================
LOG_DIR="${SCRIPT_CWD}/training_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/grid_search_${TIMESTAMP}.log"

log() { echo -e "$@" | tee -a "$MAIN_LOG"; }

log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  Grid Search training script (Batch Size + Early Stopping)${NC}"
log "${BLUE}  Start Time: $(date)${NC}"
log "${BLUE}  CUDA: ${CUDA_DEVICES}  |  Devices: ${NUM_DEVICES}${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""
log "${CYAN}[training paradigm]${NC}"
log "  1. training data auto split: train / val (val_ratio=${VAL_RATIO})"
log "  2. Batch Size grid search"
log "  3. early stopping: patience=${EARLY_STOPPING_PATIENCE}, threshold=${EARLY_STOPPING_THRESHOLD}"
log "  4. save/eval strategy: epoch (each  epoch at the endevaluationand save) "
log "  5. across BS compare best val loss → pick best → run directly test → produce final score"
log ""

# ============================================================
# ⑧ Stage 1: split training data (each dataset split only once, all model shared)
# ============================================================
log "${CYAN}[Stage 1] split training data (val_ratio=${VAL_RATIO}, seed=${VAL_SPLIT_SEED})${NC}"
log ""

mkdir -p "$SPLIT_CACHE_DIR"

for dataset in "${DATASETS[@]}"; do
    # has official val dataset: skip random split, just verify the file exists
    if [ -n "${OFFICIAL_VAL_DATA[$dataset]+_}" ]; then
        OFFICIAL_VAL="${OFFICIAL_VAL_DATA[$dataset]}"
        FULL_TRAIN="${TRIPLET_DATA_DIR}/${dataset}_triplet_training_all.json"
        if [ ! -f "$FULL_TRAIN" ]; then
            log "${RED}  ✗ ${dataset}: training file not found: ${FULL_TRAIN}${NC}"
        elif [ ! -f "$OFFICIAL_VAL" ]; then
            log "${RED}  ✗ ${dataset}: official val file not found: ${OFFICIAL_VAL}${NC}"
        else
            N_TRAIN=$(python3 -c "import json; print(len(json.load(open('$FULL_TRAIN'))))")
            N_VAL=$(python3 -c "import json; print(len(json.load(open('$OFFICIAL_VAL'))))")
            log "  ${dataset}: use official val (train=${N_TRAIN}, val=${N_VAL}) [no random split]"
        fi
        continue
    fi

    ORIG_DATA="${TRIPLET_DATA_DIR}/${dataset}_triplet_training_all.json"
    TRAIN_SPLIT="${SPLIT_CACHE_DIR}/${dataset}_train.json"
    VAL_SPLIT="${SPLIT_CACHE_DIR}/${dataset}_val.json"

    if [ ! -f "$ORIG_DATA" ]; then
        log "${RED}  ✗ data file not found: ${ORIG_DATA}, skip ${dataset}${NC}"
        continue
    fi

    if [ -f "$TRAIN_SPLIT" ] && [ -f "$VAL_SPLIT" ] && [ "$TRAIN_SPLIT" -nt "$ORIG_DATA" ]; then
        N_TRAIN=$(python3 -c "import json; print(len(json.load(open('$TRAIN_SPLIT'))))")
        N_VAL=$(python3 -c "import json; print(len(json.load(open('$VAL_SPLIT'))))")
        log "  ${dataset}: reuse existing split (train=${N_TRAIN}, val=${N_VAL})"
    else
        log "  ${dataset}: currentlyinsplit..."
        python3 "$SPLIT_SCRIPT" \
            --input "$ORIG_DATA" \
            --train-output "$TRAIN_SPLIT" \
            --val-output "$VAL_SPLIT" \
            --val-ratio "$VAL_RATIO" \
            --seed "$VAL_SPLIT_SEED" \
            2>&1 | while read line; do log "    $line"; done
    fi
done

log ""

# ============================================================
# ⑨ Stage 2: Grid Search training
# ============================================================
log "${CYAN}[Stage 2] Grid Search training${NC}"
log ""

# compute total task count
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
log "$(printf '  %-16s %-12s %-30s %-10s' 'Model' 'Dataset' 'BS Grid' 'MaxEpochs')"
log "$(printf '  %-16s %-12s %-30s %-10s' '-----' '-------' '-------' '---------')"
for model_key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    MAX_EP=${MODEL_MAX_EPOCHS[$model_key]}
    for dataset in "${DATASETS[@]}"; do
        log "$(printf '  %-16s %-12s %-30s %-10s' "$model_key" "$dataset" "${BS_GRID[$dataset]}" "$MAX_EP")"
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
        # has official val → use full training set + official val; else use the random-split result
        if [ -n "${OFFICIAL_VAL_DATA[$dataset]+_}" ]; then
            TRAIN_SPLIT="${TRIPLET_DATA_DIR}/${dataset}_triplet_training_all.json"
            VAL_SPLIT="${OFFICIAL_VAL_DATA[$dataset]}"
        else
            TRAIN_SPLIT="${SPLIT_CACHE_DIR}/${dataset}_train.json"
            VAL_SPLIT="${SPLIT_CACHE_DIR}/${dataset}_val.json"
        fi

        if [ ! -f "$TRAIN_SPLIT" ] || [ ! -f "$VAL_SPLIT" ]; then
            log "${RED}  ✗ ${dataset}: training/val file missing, skip${NC}"
            log "${RED}    train: ${TRAIN_SPLIT}${NC}"
            log "${RED}    val:   ${VAL_SPLIT}${NC}"
            for bs in ${BS_GRID[$dataset]}; do
                SKIPPED_TASKS=$((SKIPPED_TASKS + 1))
            done
            continue
        fi

        N_TRAIN=$(python3 -c "import json; print(len(json.load(open('$TRAIN_SPLIT'))))")
        N_VAL=$(python3 -c "import json; print(len(json.load(open('$VAL_SPLIT'))))")

        log ""
        log "${YELLOW}  ── ${dataset} (train=${N_TRAIN}, val=${N_VAL}) ──${NC}"

        for bs in ${BS_GRID[$dataset]}; do
            RUN_NAME="${model_key}_triplet_${dataset}_bs${bs}"
            RUN_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_NAME}"
            TASK_LOG="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"

            log ""
            log "${GREEN}  ▶ [${COMPLETED_TASKS}/${TOTAL_TASKS}] ${model_key} × ${dataset} × bs=${bs}${NC}"

            # skip already done training
            if [ -d "$RUN_OUTPUT_DIR" ]; then
                CKPT_COUNT=$(find "$RUN_OUTPUT_DIR" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)
                if [ "$CKPT_COUNT" -gt 0 ]; then
                    log "    ${CYAN}already have ${CKPT_COUNT}   checkpoint, skip training${NC}"
                    SKIPPED_TASKS=$((SKIPPED_TASKS + 1))
                    continue
                fi
            fi

            # flowvqa / flowgen_hard vision token multiple + long sequence, bs=1/GPU may still OOM;
            # use ZeRO-3 modelshard params across all cards (ZeRO-2 only shard optimizer/grad), single-card memorythen save some.
            DS_STAGE="zero3"
            case "$dataset" in
                flowvqa|flowgen_hard) DS_STAGE="zero3" ;;
            esac

            log "    epochs=${MAX_EPOCHS}  bs=${bs}  patience=${EARLY_STOPPING_PATIENCE}  deepspeed=${DS_STAGE}"
            log "    train: ${TRAIN_SPLIT}"
            log "    val:   ${VAL_SPLIT}"
            log "    out:   ${RUN_OUTPUT_DIR}"
            log "    log:   ${TASK_LOG}"

            bash "$TRAIN_SCRIPT" \
                --model_name "$MODEL_PATH" \
                --cuda_devices "$CUDA_DEVICES" \
                --data_path "$TRAIN_SPLIT" \
                --eval_data_path "$VAL_SPLIT" \
                --output_dir "$RUN_OUTPUT_DIR" \
                --num_epochs "$MAX_EPOCHS" \
                --batch_size "$bs" \
                --num_devices "$NUM_DEVICES" \
                --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
                --early_stopping_threshold "$EARLY_STOPPING_THRESHOLD" \

                --deepspeed_stage "$DS_STAGE" \
                > "$TASK_LOG" 2>&1

            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                FINAL_CKPTS=$(find "$RUN_OUTPUT_DIR" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)
                STOPPED_EPOCH=$(python3 -c "
import json, glob
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
# ⑩ Stage 3: across BS pick best → test inference → scoring
# ============================================================
log ""
# Stage 3 auto-run test inference + scoring; to train only without eval, set SKIP_INFER=1
if [ "${SKIP_INFER:-0}" = "1" ]; then
    log ""
    log "${YELLOW}SKIP_INFER=1: skipStage 3 (test inference + scoring)${NC}"
    log "${BLUE}════════════════════════════════════════════════════════════${NC}"
    log "${BLUE}  Grid Search training already done (skip inference), End: $(date)${NC}"
    log "${BLUE}════════════════════════════════════════════════════════════${NC}"
    exit 0
fi

log "${CYAN}[Stage 3] across Batch Size pick best checkpoint → test inference → scoring${NC}"
log ""

# ---- 3a. scan all run, for each (model, dataset) pick best val loss corresponding BS and checkpoint ----

# build env vars
MODELS_KEYS_STR=$(echo "${!MODELS[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')
BS_GRIDS_STR=""
for dataset in "${DATASETS[@]}"; do
    bs_list=$(echo "${BS_GRID[$dataset]}" | tr ' ' ',')
    BS_GRIDS_STR="${BS_GRIDS_STR} ${dataset}:${bs_list}"
done

# first print all BS comparison info
python3 - "$OUTPUT_BASE_DIR" "$MODELS_KEYS_STR" "${DATASETS[*]}" "$BS_GRIDS_STR" << 'PYEOF'
import json, glob, os, sys

output_base, models_str, datasets_str, bs_grids_str = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
models = models_str.split()
datasets = datasets_str.split()
bs_grids = {}
for item in bs_grids_str.split():
    ds, bss = item.split(":")
    bs_grids[ds] = bss.split(",")

for model_key in models:
    for dataset in datasets:
        runs = []
        best_val_loss = float("inf")
        best_bs = None

        for bs in bs_grids.get(dataset, []):
            run_dir = os.path.join(output_base, f"{model_key}_triplet_{dataset}_bs{bs}")
            if not os.path.isdir(run_dir):
                continue
            # scan each checkpoint trainer_state, take each one's last eval_loss
            ckpt_dirs = sorted(
                glob.glob(os.path.join(run_dir, "checkpoint-*")),
                key=lambda x: int(x.split("checkpoint-")[1])
            )
            all_evals = []
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
                continue
            best_entry = min(all_evals, key=lambda x: x[1])
            vl = best_entry[1]
            ep = best_entry[3]
            runs.append((bs, vl, ep, best_entry[0]))
            if vl < best_val_loss:
                best_val_loss = vl
                best_bs = bs

        if runs:
            print(f"  [{model_key} / {dataset}]")
            for bs, vl, ep, st in runs:
                marker = " ◀ best" if bs == best_bs else ""
                print(f"    bs={bs:>3s}  val_loss={vl:.6f}  epoch={ep}  step={st}{marker}")
            print()
PYEOF
# the above also write output tologging
python3 - "$OUTPUT_BASE_DIR" "$MODELS_KEYS_STR" "${DATASETS[*]}" "$BS_GRIDS_STR" << 'PYEOF' 2>/dev/null | while IFS= read -r line; do log "$line"; done
import json, glob, os, sys

output_base, models_str, datasets_str, bs_grids_str = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
models = models_str.split()
datasets = datasets_str.split()
bs_grids = {}
for item in bs_grids_str.split():
    ds, bss = item.split(":")
    bs_grids[ds] = bss.split(",")

for model_key in models:
    for dataset in datasets:
        runs = []
        best_val_loss = float("inf")
        best_bs = None

        for bs in bs_grids.get(dataset, []):
            run_dir = os.path.join(output_base, f"{model_key}_triplet_{dataset}_bs{bs}")
            if not os.path.isdir(run_dir):
                continue
            ckpt_dirs = sorted(
                glob.glob(os.path.join(run_dir, "checkpoint-*")),
                key=lambda x: int(x.split("checkpoint-")[1])
            )
            all_evals = []
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
                continue
            best_entry = min(all_evals, key=lambda x: x[1])
            vl = best_entry[1]
            ep = best_entry[3]
            runs.append((bs, vl, ep, best_entry[0]))
            if vl < best_val_loss:
                best_val_loss = vl
                best_bs = bs

        if runs:
            print(f"  [{model_key} / {dataset}]")
            for bs, vl, ep, st in runs:
                marker = " ◀ best" if bs == best_bs else ""
                print(f"    bs={bs:>3s}  val_loss={vl:.6f}  epoch={ep}  step={st}{marker}")
            print()
PYEOF

# ---- 3b. get best config list and run test ----
BEST_CONFIGS=$(python3 - "$OUTPUT_BASE_DIR" "$MODELS_KEYS_STR" "${DATASETS[*]}" "$BS_GRIDS_STR" << 'PYEOF'
import json, glob, os, sys

output_base, models_str, datasets_str, bs_grids_str = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
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
            run_dir = os.path.join(output_base, f"{model_key}_triplet_{dataset}_bs{bs}")
            if not os.path.isdir(run_dir):
                continue
            # scan each checkpoint trainer_state, take each one's last eval_loss
            ckpt_dirs = sorted(
                glob.glob(os.path.join(run_dir, "checkpoint-*")),
                key=lambda x: int(x.split("checkpoint-")[1])
            )
            all_evals = []
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
                continue
            best_entry = min(all_evals, key=lambda x: x[1])
            val_loss = best_entry[1]
            best_step = best_entry[0]
            best_ckpt_dir = best_entry[2]
            epoch = best_entry[3]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_info = (bs, val_loss, best_ckpt_dir, epoch, best_step)

        if best_info:
            bs, vl, cd, ep, st = best_info
            print(f"{model_key}|{dataset}|{bs}|{vl:.6f}|{cd}|{ep}|{st}")
PYEOF
)

if [ -z "$BEST_CONFIGS" ]; then
    log "${RED}no valid training results found, skip test eval${NC}"
else
    log "${CYAN}[Stage 3b] use best checkpoint run test inference + scoring${NC}"
    log ""

    mkdir -p "$EVAL_OUTPUT_DIR"
    declare -A FINAL_RESULTS

    while IFS='|' read -r model_key dataset best_bs best_val_loss best_ckpt_dir stopped_epoch best_step; do
        MODEL_PATH="${MODELS[$model_key]}"
        IMAGE_DIR="${TEST_IMAGE_DIR[$dataset]}"
        T_JSON="${TEST_JSON[$dataset]}"
        IMG_EXT="${TEST_IMAGE_EXT[$dataset]}"
        MAX_TOK="${TEST_MAX_TOKENS[$dataset]}"
        EVAL_PY="$EVAL_PY_BIN"
        EVAL_DS="${EVAL_DATASET[$dataset]}"

        RESULT_DIR="${EVAL_OUTPUT_DIR}/${model_key}"
        mkdir -p "$RESULT_DIR"
        OUTPUT_JSON="${RESULT_DIR}/${dataset}_bs${best_bs}_best.json"

        log "${GREEN}  ▶ ${model_key} / ${dataset}${NC}"
        log "    best BS:        ${best_bs}"
        log "    best val_loss:  ${best_val_loss}"
        log "    checkpoint:     ${best_ckpt_dir}"
        log "    stopped epoch:  ${stopped_epoch} (step ${best_step})"

        # -- inference --
        INFER_CMD="python -m lora.inference.get_triplets_swift_e2e"
        INFER_CMD="${INFER_CMD} --output-json ${OUTPUT_JSON}"
        INFER_CMD="${INFER_CMD} --image-dir \"${IMAGE_DIR}\""
        INFER_CMD="${INFER_CMD} --max-new-tokens ${MAX_TOK}"
        INFER_CMD="${INFER_CMD} --image-ext ${IMG_EXT}"
        INFER_CMD="${INFER_CMD} --backend local"
        INFER_CMD="${INFER_CMD} --lora-path ${best_ckpt_dir}"
        INFER_CMD="${INFER_CMD} --base-model-path ${MODEL_PATH}"
        INFER_CMD="${INFER_CMD} --gpus ${INFERENCE_GPUS}"
        if [ -n "$T_JSON" ]; then
            INFER_CMD="${INFER_CMD} --test-json ${T_JSON}"
        fi

        INFER_LOG="${LOG_DIR}/infer_${model_key}_${dataset}_bs${best_bs}_${TIMESTAMP}.log"

        if [ -f "$OUTPUT_JSON" ]; then
            log "    inference result exists, skip: ${OUTPUT_JSON}"
        else
            log "    inferring..."
            (cd "<REPO_ROOT>" && eval ${INFER_CMD}) > "${INFER_LOG}" 2>&1
            INFER_EXIT=$?

            if [ $INFER_EXIT -ne 0 ]; then
                log "    ${RED}✗ inference failed (exit=${INFER_EXIT}), see: ${INFER_LOG}${NC}"
                continue
            fi
            log "    inference done → ${OUTPUT_JSON}"
        fi

        # -- scoring --
        if [ ! -f "$EVAL_PY" ]; then
            log "    ${RED}✗ eval script not found: ${EVAL_PY}${NC}"
            continue
        fi

        log "    scoring..."
        EVAL_OUTPUT=$(cd "<REPO_ROOT>" && python "$EVAL_PY" --dataset "$EVAL_DS" --prediction-file "$OUTPUT_JSON" --image-dir "$IMAGE_DIR" --test-json "${T_JSON:-}" 2>&1)
        EVAL_EXIT=$?

        if [ $EVAL_EXIT -ne 0 ]; then
            log "    ${RED}✗ scoring failed${NC}"
            log "    ${EVAL_OUTPUT}"
            continue
        fi

        # extract F1 score
        SCORE=$(echo "$EVAL_OUTPUT" | python3 -c "
import sys, re, json, os
text = sys.stdin.read()
# first try from evaluation results JSON read from file
output_json = '${OUTPUT_JSON}'
result_dir = os.path.dirname(output_json)
stem = os.path.splitext(os.path.basename(output_json))[0]
for fname in os.listdir(result_dir):
    if fname.startswith('evaluation_results') and fname.endswith('.json') and stem in fname:
        with open(os.path.join(result_dir, fname)) as f:
            data = json.load(f)
        if 'f1_score' in data:
            print(f'{data[\"f1_score\"]:.4f}')
            sys.exit(0)
# fallback: from stdout regex match
for pat in [r'Relaxed F1[:\s]+([0-9.]+)', r'rF1[:\s]+([0-9.]+)',
            r'F1 Score[:\s]+([0-9.]+)', r'F1[:\s]+([0-9.]+)',
            r'f1[:\s]+([0-9.]+)', r'score[:\s]+([0-9.]+)']:
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        print(m.group(1))
        sys.exit(0)
print('N/A')
" 2>/dev/null)

        log "    ${GREEN}✓ F1 Score = ${SCORE}${NC}"
        log ""

        FINAL_RESULTS["${model_key}|${dataset}"]="${best_bs}|${best_val_loss}|${SCORE}|${stopped_epoch}|${best_step}"

    done <<< "$BEST_CONFIGS"

    # ---- 3c. final summary table ----
    log ""
    log "${BLUE}════════════════════════════════════════════════════════════${NC}"
    log "${BLUE}  final results summary${NC}"
    log "${BLUE}════════════════════════════════════════════════════════════${NC}"
    log ""
    log "$(printf '  %-16s %-12s %-6s %-12s %-12s %-8s' 'Model' 'Dataset' 'BS' 'ValLoss' 'TestF1' 'Epoch')"
    log "$(printf '  %-16s %-12s %-6s %-12s %-12s %-8s' '-----' '-------' '--' '-------' '------' '-----')"

    for key in $(echo "${!FINAL_RESULTS[@]}" | tr ' ' '\n' | sort); do
        IFS='|' read -r model_key dataset <<< "$key"
        IFS='|' read -r best_bs best_val_loss score stopped_epoch best_step <<< "${FINAL_RESULTS[$key]}"
        log "$(printf '  %-16s %-12s %-6s %-12s %-12s %-8s' "$model_key" "$dataset" "$best_bs" "$best_val_loss" "$score" "$stopped_epoch")"
    done

    # save results JSON
    SUMMARY_JSON="${EVAL_OUTPUT_DIR}/grid_search_results_${TIMESTAMP}.json"
    python3 -c "
import json
results = {}
raw = '''$(for key in "${!FINAL_RESULTS[@]}"; do echo "${key}=${FINAL_RESULTS[$key]}"; done)'''
for line in raw.strip().split('\n'):
    if not line.strip():
        continue
    k, v = line.split('=', 1)
    model, dataset = k.split('|')
    bs, val_loss, score, epoch, step = v.split('|')
    results[f'{model}/{dataset}'] = {
        'best_batch_size': int(bs),
        'best_val_loss': float(val_loss),
        'test_f1_score': score,
        'stopped_epoch': epoch,
        'best_step': step,
    }
with open('$SUMMARY_JSON', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'results saved: $SUMMARY_JSON')
" 2>&1 | while read line; do log "  $line"; done
fi

# ============================================================
# ⑪ summary
# ============================================================
log ""
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  all done!  End Time: $(date)${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""
log "  training tasks:  ${TOTAL_TASKS}"
log "  ${GREEN}done:    ${COMPLETED_TASKS}${NC}"
log "  ${CYAN}skip:    ${SKIPPED_TASKS}${NC}"
log "  ${RED}failed:    ${FAILED_TASKS}${NC}"
log ""
log "  main log:    ${MAIN_LOG}"
log "  sub log:    ${LOG_DIR}/"
log "  training output:  ${OUTPUT_BASE_DIR}/"
log "  eval output:  ${EVAL_OUTPUT_DIR}/"

if [ $FAILED_TASKS -gt 0 ]; then
    log ""
    log "${RED}⚠  has ${FAILED_TASKS}  training tasksfailed${NC}"
    exit 1
else
    log ""
    log "${GREEN}✅ all done${NC}"
    exit 0
fi
