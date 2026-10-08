#!/bin/bash

# ============================================================
# train_all_E2E_grid_search.sh (MiniCPM-V4.5-8B, SWIFT)
#
# Triplet version Grid Search - structure vs Qwen version aligned, training layer switched to SWIFT.
#
# vs Arrow version difference:
# - training data as triplet format ({dataset}_triplet_training_all.json)
# - inference uses get_triplets_sft_e2e (whole-image single output JSON)
# - evaluation script is lora/eval/eval_E2E.py
#
# Usage:
# cd MLLMs_SFT/MiniCPM-V4.5-8B
# bash scripts/train_all_E2E_grid_search.sh
# ============================================================

SCRIPT_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    ["minicpm_v4.5"]="<MODEL_ROOT>/MiniCPM-V-4_5"
)

declare -A MODEL_MAX_EPOCHS
MODEL_MAX_EPOCHS=(
    ["minicpm_v4.5"]=10
)

# ============================================================
# ② Dataset config
# ============================================================
DATASETS=("flowlearn" "flowvqa" "fcb" "fca" "cbd" "bpmn" "flowgen_easy" "flowgen_medium" "flowgen_hard")

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
SWIFT_DATA_CACHE="${SCRIPT_CWD}/output/_swift_data_cache"
EVAL_OUTPUT_DIR="<OUTPUT_ROOT>/output_lora_triplet"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_early_stop.sh"
SPLIT_SCRIPT="${SCRIPT_DIR}/split_train_val.py"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_to_swift_format.py"

# official val data
declare -A OFFICIAL_VAL_DATA
OFFICIAL_VAL_DATA=(
    ["cbd"]="${TRIPLET_DATA_DIR}/cbd_val_triplet_training_all.json"
    ["fcb"]="${TRIPLET_DATA_DIR}/fcb_val_triplet_training_all.json"
    ["bpmn"]="${TRIPLET_DATA_DIR}/bpmn_triplet_dev.json"
)

# ============================================================
# ⑤-b test set config (Stage 3 inference + scoring)
# ============================================================
declare -A TEST_IMAGE_DIR
TEST_IMAGE_DIR=(
    ["fca"]="<DATA_ROOT>/Block-Diagram-Datasets/Handwritten_block_diagrams/FC_A/test_img"
    ["cbd"]="<DATA_ROOT>/Block-Diagram-Datasets/Computerized_block_diagrams(CBD)/test_img"
    ["fcb"]="<DATA_ROOT>/Block-Diagram-Datasets/Handwritten_block_diagrams/FC_B/test"
    ["flowlearn"]="<DATA_ROOT>/flowlearn/mermaid_word/jpeg"
    ["flowvqa"]="<DATA_ROOT>/flowvqa/Data/A. Main Set Flowchart Images"
    ["bpmn"]="<DATA_ROOT>/bpmn/test_images"
    ["flowgen_easy"]="<DATA_ROOT>/flowgen/test_img_easy"
    ["flowgen_medium"]="<DATA_ROOT>/flowgen/test_img_medium"
    ["flowgen_hard"]="<DATA_ROOT>/flowgen/test_img_hard"
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
MAIN_LOG="${LOG_DIR}/triplet_grid_search_${TIMESTAMP}.log"

log() { echo -e "$@" | tee -a "$MAIN_LOG"; }

log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  MiniCPM-V4.5 Triplet Grid Search (SWIFT + Early Stopping)${NC}"
log "${BLUE}  Start Time: $(date)${NC}"
log "${BLUE}  CUDA: ${CUDA_DEVICES}  |  Devices: ${NUM_DEVICES}${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""

# ============================================================
# ⑧ Stage 1: prepare data
# ============================================================
log "${CYAN}[Stage 1] prepare data (LLaVA → SWIFT format conversion + val)${NC}"
log ""

mkdir -p "$SPLIT_CACHE_DIR" "$SWIFT_DATA_CACHE"

convert_if_needed() {
    local src="$1"
    local dst="$2"
    if [ -f "$dst" ] && [ "$dst" -nt "$src" ]; then
        return 0
    fi
    python3 "$CONVERT_SCRIPT" --input "$src" --output "$dst"
}

for dataset in "${DATASETS[@]}"; do
    # has official val → use full train + official val
    if [ -n "${OFFICIAL_VAL_DATA[$dataset]+_}" ]; then
        OFFICIAL_VAL="${OFFICIAL_VAL_DATA[$dataset]}"
        FULL_TRAIN="${TRIPLET_DATA_DIR}/${dataset}_triplet_training_all.json"
        if [ ! -f "$FULL_TRAIN" ]; then
            log "${RED}  ✗ ${dataset}: training file not found: ${FULL_TRAIN}${NC}"
            continue
        elif [ ! -f "$OFFICIAL_VAL" ]; then
            log "${RED}  ✗ ${dataset}: official val not found: ${OFFICIAL_VAL}${NC}"
            continue
        fi
        SWIFT_TRAIN="${SWIFT_DATA_CACHE}/${dataset}_triplet_train_swift.json"
        SWIFT_VAL="${SWIFT_DATA_CACHE}/${dataset}_triplet_val_swift.json"
        convert_if_needed "$FULL_TRAIN" "$SWIFT_TRAIN"
        convert_if_needed "$OFFICIAL_VAL" "$SWIFT_VAL"
        ln -sf "$SWIFT_TRAIN" "${SPLIT_CACHE_DIR}/${dataset}_train.json"
        ln -sf "$SWIFT_VAL" "${SPLIT_CACHE_DIR}/${dataset}_val.json"
        N_TRAIN=$(python3 -c "import json; print(len(json.load(open('$SWIFT_TRAIN'))))")
        N_VAL=$(python3 -c "import json; print(len(json.load(open('$SWIFT_VAL'))))")
        log "  ${dataset}: official val (train=${N_TRAIN}, val=${N_VAL})"
        continue
    fi

    # no official val → from train split
    ORIG_DATA="${TRIPLET_DATA_DIR}/${dataset}_triplet_training_all.json"
    TRAIN_SPLIT="${SPLIT_CACHE_DIR}/${dataset}_train_llava.json"
    VAL_SPLIT="${SPLIT_CACHE_DIR}/${dataset}_val_llava.json"

    if [ ! -f "$ORIG_DATA" ]; then
        log "${RED}  ✗ ${dataset}: data not found: ${ORIG_DATA}${NC}"
        continue
    fi

    if [ -f "$TRAIN_SPLIT" ] && [ -f "$VAL_SPLIT" ] && [ "$TRAIN_SPLIT" -nt "$ORIG_DATA" ]; then
        log "  ${dataset}: reuse existing split"
    else
        log "  ${dataset}: splitin..."
        python3 "$SPLIT_SCRIPT" \
            --input "$ORIG_DATA" \
            --train-output "$TRAIN_SPLIT" \
            --val-output "$VAL_SPLIT" \
            --val-ratio "$VAL_RATIO" \
            --seed "$VAL_SPLIT_SEED" \
            2>&1 | while read line; do log "    $line"; done
    fi
    SWIFT_TRAIN="${SWIFT_DATA_CACHE}/${dataset}_triplet_train_swift.json"
    SWIFT_VAL="${SWIFT_DATA_CACHE}/${dataset}_triplet_val_swift.json"
    convert_if_needed "$TRAIN_SPLIT" "$SWIFT_TRAIN"
    convert_if_needed "$VAL_SPLIT" "$SWIFT_VAL"
    ln -sf "$SWIFT_TRAIN" "${SPLIT_CACHE_DIR}/${dataset}_train.json"
    ln -sf "$SWIFT_VAL" "${SPLIT_CACHE_DIR}/${dataset}_val.json"
    N_TRAIN=$(python3 -c "import json; print(len(json.load(open('$SWIFT_TRAIN'))))")
    N_VAL=$(python3 -c "import json; print(len(json.load(open('$SWIFT_VAL'))))")
    log "  ${dataset}: randomsplit (train=${N_TRAIN}, val=${N_VAL})"
done

log ""

# ============================================================
# ⑨ Stage 2: Grid Search training
# ============================================================
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    log "${YELLOW}[Stage 2] skip training (SKIP_TRAIN=1), go directly toStage 3 eval${NC}"
    log ""
    COMPLETED_TASKS=0; FAILED_TASKS=0; SKIPPED_TASKS=0; TOTAL_TASKS=0
else

log "${CYAN}[Stage 2] Grid Search training (Triplet, SWIFT)${NC}"
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

for model_key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    MODEL_PATH="${MODELS[$model_key]}"
    MAX_EPOCHS="${MODEL_MAX_EPOCHS[$model_key]}"

    log "${BLUE}════════════════════════════════════════════════════════════${NC}"
    log "${BLUE}  model: ${model_key}  (max_epochs=${MAX_EPOCHS})${NC}"
    log "${BLUE}════════════════════════════════════════════════════════════${NC}"

    for dataset in "${DATASETS[@]}"; do
        TRAIN_DATA="${SPLIT_CACHE_DIR}/${dataset}_train.json"
        VAL_DATA="${SPLIT_CACHE_DIR}/${dataset}_val.json"

        [ -L "$TRAIN_DATA" ] && TRAIN_DATA=$(readlink -f "$TRAIN_DATA")
        [ -L "$VAL_DATA" ] && VAL_DATA=$(readlink -f "$VAL_DATA")

        if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
            log "${RED}  ✗ ${dataset}: data missing, skip${NC}"
            for bs in ${BS_GRID[$dataset]}; do SKIPPED_TASKS=$((SKIPPED_TASKS + 1)); done
            continue
        fi

        N_TRAIN=$(python3 -c "import json; print(len(json.load(open('$TRAIN_DATA'))))")
        N_VAL=$(python3 -c "import json; print(len(json.load(open('$VAL_DATA'))))")

        log ""
        log "${YELLOW}  ── ${dataset} (train=${N_TRAIN}, val=${N_VAL}) ──${NC}"

        for bs in ${BS_GRID[$dataset]}; do
            RUN_NAME="${model_key}_triplet_${dataset}_bs${bs}"
            RUN_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_NAME}"
            TASK_LOG="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"

            log ""
            log "${GREEN}  ▶ [${COMPLETED_TASKS}/${TOTAL_TASKS}] ${model_key} × ${dataset} × bs=${bs}${NC}"

            if [ -d "$RUN_OUTPUT_DIR" ]; then
                CKPT_COUNT=$(find "$RUN_OUTPUT_DIR" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)
                if [ "$CKPT_COUNT" -gt 0 ]; then
                    log "    ${CYAN}already have ${CKPT_COUNT}   checkpoint, will resume${NC}"
                fi
            fi

            log "    epochs=${MAX_EPOCHS}  bs=${bs}  patience=${EARLY_STOPPING_PATIENCE}"
            log "    out: ${RUN_OUTPUT_DIR}"

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
state_file = '${RUN_OUTPUT_DIR}/trainer_state.json'
if os.path.isfile(state_file):
    with open(state_file) as f: state = json.load(f)
    print(state.get('epoch', '?'))
else:
    fs = sorted(glob.glob('${RUN_OUTPUT_DIR}/checkpoint-*/trainer_state.json'),
                key=lambda x: int(x.split('checkpoint-')[1].split('/')[0]))
    if fs:
        with open(fs[-1]) as f: state = json.load(f)
        print(state.get('epoch', '?'))
    else: print('?')
" 2>/dev/null)
                log "    ${GREEN}✓ success (epoch=${STOPPED_EPOCH}, ${FINAL_CKPTS} checkpoints)${NC}"
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

fi  # end of SKIP_TRAIN guard

# ============================================================
# ⑩ Stage 3: across BS pick best → test inference → scoring
# ============================================================
log ""
log "${CYAN}[Stage 3] across BS pick best checkpoint → test inference → scoring${NC}"
log ""

MODELS_KEYS_STR=$(echo "${!MODELS[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')
BS_GRIDS_STR=""
for dataset in "${DATASETS[@]}"; do
    bs_list=$(echo "${BS_GRID[$dataset]}" | tr ' ' ',')
    BS_GRIDS_STR="${BS_GRIDS_STR} ${dataset}:${bs_list}"
done

# print BS compare
python3 - "$OUTPUT_BASE_DIR" "$MODELS_KEYS_STR" "${DATASETS[*]}" "$BS_GRIDS_STR" "triplet" << 'PYEOF' | while IFS= read -r line; do log "$line"; done
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
        runs = []
        best_val_loss = float("inf")
        best_bs = None
        for bs in bs_grids.get(dataset, []):
            run_dir = os.path.join(output_base, f"{model_key}_{task_type}_{dataset}_bs{bs}")
            if not os.path.isdir(run_dir): continue
            state_file = os.path.join(run_dir, "trainer_state.json")
            if os.path.isfile(state_file):
                with open(state_file) as f: state = json.load(f)
            else:
                # compatible Qwen flat (checkpoint-*) and MiniCPM/SWIFT nested (v*-*/checkpoint-*)
                state_files = sorted(
                    glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
                    + glob.glob(os.path.join(run_dir, "v*-*", "checkpoint-*", "trainer_state.json")),
                    key=lambda x: int(x.rsplit("checkpoint-", 1)[1].split("/")[0]))
                if not state_files: continue
                with open(state_files[-1]) as f: state = json.load(f)
            eval_entries = [e for e in state.get("log_history", []) if "eval_loss" in e]
            if not eval_entries: continue
            best_entry = min(eval_entries, key=lambda e: e["eval_loss"])
            vl = best_entry["eval_loss"]
            ep = state.get("epoch", "?")
            runs.append((bs, vl, ep, best_entry.get("step", "?")))
            if vl < best_val_loss: best_val_loss = vl; best_bs = bs
        if runs:
            print(f"  [{model_key} / {dataset}]")
            for bs, vl, ep, st in runs:
                marker = " ◀ best" if bs == best_bs else ""
                print(f"    bs={bs:>3s}  val_loss={vl:.6f}  epoch={ep}  step={st}{marker}")
            print()
PYEOF

# get best config
BEST_CONFIGS=$(python3 - "$OUTPUT_BASE_DIR" "$MODELS_KEYS_STR" "${DATASETS[*]}" "$BS_GRIDS_STR" "triplet" << 'PYEOF'
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
            if not os.path.isdir(run_dir): continue
            state_file = os.path.join(run_dir, "trainer_state.json")
            best_ckpt_for_state = None
            if os.path.isfile(state_file):
                with open(state_file) as f: state = json.load(f)
            else:
                # compatible Qwen flat (checkpoint-*) and MiniCPM/SWIFT nested (v*-*/checkpoint-*)
                state_files = sorted(
                    glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
                    + glob.glob(os.path.join(run_dir, "v*-*", "checkpoint-*", "trainer_state.json")),
                    key=lambda x: int(x.rsplit("checkpoint-", 1)[1].split("/")[0]))
                if not state_files: continue
                with open(state_files[-1]) as f: state = json.load(f)
                best_ckpt_for_state = state_files[-1].rsplit("/", 1)[0]
            eval_entries = [e for e in state.get("log_history", []) if "eval_loss" in e]
            if not eval_entries: continue
            best_entry = min(eval_entries, key=lambda e: e["eval_loss"])
            val_loss = best_entry["eval_loss"]
            best_step = best_entry.get("step", 0)
            epoch = state.get("epoch", "?")
            if best_ckpt_for_state:
                best_ckpt_dir = best_ckpt_for_state
            else:
                best_ckpt_dir = os.path.join(run_dir, f"checkpoint-{best_step}")
                if not os.path.isdir(best_ckpt_dir):
                    # compatible with nested v*-*/checkpoint-N
                    nested = glob.glob(os.path.join(run_dir, "v*-*", f"checkpoint-{best_step}"))
                    if nested and os.path.isdir(nested[0]):
                        best_ckpt_dir = nested[0]
                    else:
                        ckpt_dirs = sorted(
                            glob.glob(os.path.join(run_dir, "checkpoint-*"))
                            + glob.glob(os.path.join(run_dir, "v*-*", "checkpoint-*")),
                            key=lambda x: int(x.rsplit("checkpoint-", 1)[1]))
                        best_ckpt_dir = ckpt_dirs[-1] if ckpt_dirs else run_dir
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_info = (bs, val_loss, best_ckpt_dir, epoch, best_step)
        if best_info:
            bs, vl, cd, ep, st = best_info
            print(f"{model_key}|{dataset}|{bs}|{vl:.6f}|{cd}|{ep}|{st}")
PYEOF
)

if [ -z "$BEST_CONFIGS" ]; then
    log "${RED}no valid foundtraining result, skip test${NC}"
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

        # -- inference (MiniCPM-V4.5 dedicated: AutoModel + model.chat) --
        INFER_CMD="python -m lora.inference.get_triplets_sft_e2e_minicpm"
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

        INFER_LOG="${LOG_DIR}/infer_triplet_${model_key}_${dataset}_bs${best_bs}_${TIMESTAMP}.log"

        if [ -f "$OUTPUT_JSON" ]; then
            log "    inference result exists, skip: ${OUTPUT_JSON}"
        else
            log "    inferring..."
            (cd "<REPO_ROOT>" && eval ${INFER_CMD}) > "${INFER_LOG}" 2>&1
            INFER_EXIT=$?
            if [ $INFER_EXIT -ne 0 ]; then
                log "    ${RED}✗ inference failed (exit=${INFER_EXIT})${NC}"
                tail -5 "$INFER_LOG" 2>/dev/null | while read line; do log "      $line"; done
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

        SCORE=$(echo "$EVAL_OUTPUT" | python3 -c "
import sys, re, json, os
text = sys.stdin.read()
output_json = '${OUTPUT_JSON}'
result_dir = os.path.dirname(output_json)
stem = os.path.splitext(os.path.basename(output_json))[0]
for fname in os.listdir(result_dir):
    if fname.startswith('evaluation_results') and fname.endswith('.json') and stem in fname:
        with open(os.path.join(result_dir, fname)) as f: data = json.load(f)
        if 'f1_score' in data: print(f'{data[\"f1_score\"]:.4f}'); sys.exit(0)
for pat in [r'Relaxed F1[:\s]+([0-9.]+)', r'rF1[:\s]+([0-9.]+)',
            r'F1 Score[:\s]+([0-9.]+)', r'F1[:\s]+([0-9.]+)']:
    m = re.search(pat, text, re.IGNORECASE)
    if m: print(m.group(1)); sys.exit(0)
print('N/A')
" 2>/dev/null)

        log "    ${GREEN}✓ F1 Score = ${SCORE}${NC}"
        log ""
        FINAL_RESULTS["${model_key}|${dataset}"]="${best_bs}|${best_val_loss}|${SCORE}|${stopped_epoch}|${best_step}"

    done <<< "$BEST_CONFIGS"

    # summary
    log ""
    log "${BLUE}════════════════════════════════════════════════════════════${NC}"
    log "${BLUE}  Triplet final results summary (MiniCPM-V4.5)${NC}"
    log "${BLUE}════════════════════════════════════════════════════════════${NC}"
    log ""
    log "$(printf '  %-20s %-16s %-6s %-12s %-12s %-8s' 'Model' 'Dataset' 'BS' 'ValLoss' 'TestF1' 'Epoch')"
    log "$(printf '  %-20s %-16s %-6s %-12s %-12s %-8s' '-----' '-------' '--' '-------' '------' '-----')"

    for key in $(echo "${!FINAL_RESULTS[@]}" | tr ' ' '\n' | sort); do
        IFS='|' read -r model_key dataset <<< "$key"
        IFS='|' read -r best_bs best_val_loss score stopped_epoch best_step <<< "${FINAL_RESULTS[$key]}"
        log "$(printf '  %-20s %-16s %-6s %-12s %-12s %-8s' "$model_key" "$dataset" "$best_bs" "$best_val_loss" "$score" "$stopped_epoch")"
    done

    SUMMARY_JSON="${EVAL_OUTPUT_DIR}/minicpm_triplet_grid_search_results_${TIMESTAMP}.json"
    python3 -c "
import json
results = {}
raw = '''$(for key in \"\${!FINAL_RESULTS[@]}\"; do echo \"\${key}=\${FINAL_RESULTS[\$key]}\"; done)'''
for line in raw.strip().split('\n'):
    if not line.strip(): continue
    k, v = line.split('=', 1)
    model, dataset = k.split('|')
    bs, val_loss, score, epoch, step = v.split('|')
    results[f'{model}/{dataset}'] = {
        'best_batch_size': int(bs), 'best_val_loss': float(val_loss),
        'test_f1_score': score, 'stopped_epoch': epoch, 'best_step': step,
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
log "${BLUE}  MiniCPM Triplet Grid Search done!  End Time: $(date)${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""
log "  training: ${TOTAL_TASKS}  done: ${COMPLETED_TASKS}  skip: ${SKIPPED_TASKS}  failed: ${FAILED_TASKS}"
log "  logging: ${MAIN_LOG}"
log "  output: ${OUTPUT_BASE_DIR}/"

if [ $FAILED_TASKS -gt 0 ]; then
    exit 1
else
    log "${GREEN}✅ all done${NC}"
    exit 0
fi
