#!/bin/bash

# ============================================================
# eval_baseline_no_sft.sh (MiniCPM-V4.5)
#
# no training baseline evaluation script (MiniCPM-V-4_5 base model)
#
# Function:
# to MiniCPM-V-4_5 base model (no LoRA) evaluation triplet and arrow two modes
# - Triplet: get_triplets_sft_e2e_minicpm (whole-image, no --lora-path)
# - Arrow: get_triplets_sft_trace_minicpm (per-arrow, SAM3 / GT, no --adapter-path)
#
# Usage:
# cd MLLMs_SFT/MiniCPM-V4.5
# bash scripts/eval_baseline_no_sft.sh
#
# ============================================================

SCRIPT_CWD="$(pwd)"

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

# ============================================================
# ② dataset / mode config
# ============================================================
DATASETS=("bpmn")

# evaluation mode: triplet / arrow / run both
EVAL_MODES=("triplet" "arrow")

# Arrow mode arrow source
# - sam: SAM3 pretraining (no finetune)
# - sam3_ft: SAM3 fine-tuned detector (baseline usually unused)
# - groundtruth: GT arrow annotation (diagnostic / oracle)
ARROW_SOURCES="sam"

# ============================================================
# ③ Path config
# ============================================================
EVAL_OUTPUT_DIR="<OUTPUT_ROOT>/output_baseline_no_sft_minicpm"
INFERENCE_GPUS="0,1,2,3,4,5,6,7"

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
    ["flowgen_easy_orig"]="<DATA_ROOT>/flowgen/test_img_easy_orig"
    ["flowgen_medium_orig"]="<DATA_ROOT>/flowgen/test_img_medium_orig"
    ["flowgen_hard_orig"]="<DATA_ROOT>/flowgen/test_img_hard_orig"
)

declare -A TEST_JSON
TEST_JSON=(
    ["fca"]=""
    ["cbd"]=""
    ["fcb"]=""
    ["flowlearn"]="<DATA_ROOT>/flowlearn/test.json"
    ["flowvqa"]="<DATA_ROOT>/flowvqa/Data/test_full.json"
    ["bpmn"]=""
    ["flowgen_easy"]=""
    ["flowgen_medium"]=""
    ["flowgen_hard"]=""
    ["flowgen_easy_orig"]=""
    ["flowgen_medium_orig"]=""
    ["flowgen_hard_orig"]=""
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
    ["flowgen_easy_orig"]=".png"
    ["flowgen_medium_orig"]=".png"
    ["flowgen_hard_orig"]=".png"
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
    ["flowgen_easy_orig"]="4096"
    ["flowgen_medium_orig"]="8192"
    ["flowgen_hard_orig"]="8192"
)

EVAL_PY_TRIPLET="<REPO_ROOT>/lora/eval/eval_E2E.py"
EVAL_PY_ARROW="<REPO_ROOT>/lora/eval/eval_TRACE.py"
declare -A EVAL_DATASET
EVAL_DATASET=(
    ["fca"]="fca" ["cbd"]="cbd" ["fcb"]="fcb"
    ["flowlearn"]="flowlearn" ["flowvqa"]="flowvqa" ["bpmn"]="bpmn"
    ["flowgen_easy"]="flowgen" ["flowgen_medium"]="flowgen" ["flowgen_hard"]="flowgen"
)

declare -A EVAL_GT_ROOT
EVAL_GT_ROOT=(
    ["flowgen_easy"]="<DATA_ROOT>/data_4_training/flowgen_test_easy"
    ["flowgen_medium"]="<DATA_ROOT>/data_4_training/flowgen_test_medium"
    ["flowgen_hard"]="<DATA_ROOT>/data_4_training/flowgen_test_hard"
)

# ============================================================
# ④ Logging config
# ============================================================
LOG_DIR="${SCRIPT_CWD}/training_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/baseline_no_sft_minicpm_${TIMESTAMP}.log"

log() { echo -e "$@" | tee -a "$MAIN_LOG"; }

log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  MiniCPM-V4.5 Baseline (no training) evaluation script${NC}"
log "${BLUE}  Models:        ${!MODELS[*]}${NC}"
log "${BLUE}  Datasets:      ${DATASETS[*]}${NC}"
log "${BLUE}  Modes:         ${EVAL_MODES[*]}${NC}"
log "${BLUE}  Arrow Sources: ${ARROW_SOURCES}${NC}"
log "${BLUE}  Start Time:    $(date)${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""

mkdir -p "$EVAL_OUTPUT_DIR"

# ============================================================
# ⑤ extract F1 score generic function
# ============================================================
extract_f1() {
    local output_json="$1"
    local eval_output="$2"
    python3 -c "
import sys, re, json, os
text = '''${eval_output}'''
output_json = '${output_json}'
result_dir = os.path.dirname(output_json)
stem = os.path.splitext(os.path.basename(output_json))[0]
if os.path.isdir(result_dir):
    for fname in os.listdir(result_dir):
        if fname.startswith('evaluation_results') and fname.endswith('.json') and stem in fname:
            with open(os.path.join(result_dir, fname)) as f:
                data = json.load(f)
            if 'f1_score' in data:
                print(f'{data[\"f1_score\"]:.4f}')
                sys.exit(0)
for pat in [r'Relaxed F1[:\s]+([0-9.]+)', r'rF1[:\s]+([0-9.]+)',
            r'F1 Score[:\s]+([0-9.]+)', r'F1[:\s]+([0-9.]+)',
            r'f1[:\s]+([0-9.]+)', r'score[:\s]+([0-9.]+)']:
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        print(m.group(1))
        sys.exit(0)
print('N/A')
" 2>/dev/null
}

# ============================================================
# ⑥ main loop: iterate model × dataset × mode
# ============================================================
declare -A FINAL_RESULTS
TASK_IDX=0
TOTAL_TASKS=0

for model_key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    for dataset in "${DATASETS[@]}"; do
        for mode in "${EVAL_MODES[@]}"; do
            if [[ "$mode" == "arrow" ]]; then
                for arrow_src in $ARROW_SOURCES; do
                    TOTAL_TASKS=$((TOTAL_TASKS + 1))
                done
            else
                TOTAL_TASKS=$((TOTAL_TASKS + 1))
            fi
        done
    done
done

log "total tasks: ${TOTAL_TASKS}"
log ""

for model_key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    MODEL_PATH="${MODELS[$model_key]}"

    for dataset in "${DATASETS[@]}"; do
        IMAGE_DIR="${TEST_IMAGE_DIR[$dataset]}"
        T_JSON="${TEST_JSON[$dataset]}"
        IMG_EXT="${TEST_IMAGE_EXT[$dataset]}"
        MAX_TOK="${TEST_MAX_TOKENS[$dataset]}"

        for mode in "${EVAL_MODES[@]}"; do

            if [[ "$mode" == "triplet" ]]; then
                # ── Triplet mode ──
                EVAL_PY="$EVAL_PY_TRIPLET"
                EVAL_DS="${EVAL_DATASET[$dataset]}"
                RESULT_DIR="${EVAL_OUTPUT_DIR}/${model_key}"
                mkdir -p "$RESULT_DIR"
                OUTPUT_JSON="${RESULT_DIR}/${dataset}_baseline_triplet.json"

                log "${GREEN}  ▶ [${TASK_IDX}/${TOTAL_TASKS}] ${model_key} / ${dataset} / triplet${NC}"

                INFER_LOG="${LOG_DIR}/infer_baseline_${model_key}_${dataset}_triplet_${TIMESTAMP}.log"

                if [ -f "$OUTPUT_JSON" ]; then
                    log "    inference result exists, skip: ${OUTPUT_JSON}"
                else
                    log "    inferring (triplet, base model, no LoRA)..."
                    INFER_CMD="python -m lora.inference.get_triplets_sft_e2e_minicpm"
                    INFER_CMD="${INFER_CMD} --output-json ${OUTPUT_JSON}"
                    INFER_CMD="${INFER_CMD} --image-dir \"${IMAGE_DIR}\""
                    INFER_CMD="${INFER_CMD} --max-new-tokens ${MAX_TOK}"
                    INFER_CMD="${INFER_CMD} --image-ext ${IMG_EXT}"
                    INFER_CMD="${INFER_CMD} --backend local"
                    INFER_CMD="${INFER_CMD} --base-model-path ${MODEL_PATH}"
                    INFER_CMD="${INFER_CMD} --gpus ${INFERENCE_GPUS}"
                    INFER_CMD="${INFER_CMD} --workers-per-gpu 2"
                    if [ -n "$T_JSON" ]; then
                        INFER_CMD="${INFER_CMD} --test-json ${T_JSON}"
                    fi

                    (cd "<REPO_ROOT>" && eval ${INFER_CMD}) > "${INFER_LOG}" 2>&1
                    INFER_EXIT=$?

                    if [ $INFER_EXIT -ne 0 ]; then
                        log "    ${RED}✗ inference failed (exit=${INFER_EXIT}), see: ${INFER_LOG}${NC}"
                        TASK_IDX=$((TASK_IDX + 1))
                        continue
                    fi
                    log "    inference done → ${OUTPUT_JSON}"
                fi

                if [ ! -f "$EVAL_PY" ]; then
                    log "    ${RED}✗ eval script not found: ${EVAL_PY}${NC}"
                    TASK_IDX=$((TASK_IDX + 1))
                    continue
                fi

                log "    scoring..."
                EVAL_OUTPUT=$(cd "<REPO_ROOT>" && python "$EVAL_PY" --dataset "$EVAL_DS" --prediction-file "$OUTPUT_JSON" --image-dir "$IMAGE_DIR" --test-json "${T_JSON:-}" 2>&1)
                EVAL_EXIT=$?

                if [ $EVAL_EXIT -ne 0 ]; then
                    log "    ${RED}✗ scoring failed${NC}"
                    log "    ${EVAL_OUTPUT}"
                    TASK_IDX=$((TASK_IDX + 1))
                    continue
                fi

                SCORE=$(extract_f1 "$OUTPUT_JSON" "$EVAL_OUTPUT")
                log "    ${GREEN}✓ F1 Score = ${SCORE}${NC}"
                log ""

                FINAL_RESULTS["${model_key}|${dataset}|triplet"]="${SCORE}"
                TASK_IDX=$((TASK_IDX + 1))

            elif [[ "$mode" == "arrow" ]]; then
                # ── Arrow mode (loop arrow sources) ──
                for ARROW_SRC in $ARROW_SOURCES; do
                    EVAL_PY="$EVAL_PY_ARROW"
                    EVAL_DS="${EVAL_DATASET[$dataset]}"
                    OUTPUT_DIR_ARROW="${EVAL_OUTPUT_DIR}/${model_key}_baseline_${dataset}_${ARROW_SRC}"
                    mkdir -p "$OUTPUT_DIR_ARROW"

                    log "${GREEN}  ▶ [${TASK_IDX}/${TOTAL_TASKS}] ${model_key} / ${dataset} / arrow / ${ARROW_SRC}${NC}"

                    INFER_LOG="${LOG_DIR}/infer_baseline_${model_key}_${dataset}_arrow_${ARROW_SRC}_${TIMESTAMP}.log"

                    EXISTING_COUNT=$(find "$OUTPUT_DIR_ARROW" -name "arrow_triplets.json" 2>/dev/null | wc -l)
                    if [ -n "$T_JSON" ] && [ -f "$T_JSON" ]; then
                        # only count test-json listed in sample (IMAGE_DIR other images in dir won't run)
                        TOTAL_IMAGES=$(python3 -c "
import json
with open('$T_JSON', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(len(data) if isinstance(data, (list, dict)) else 0)
" 2>/dev/null)
                        if [ -z "$TOTAL_IMAGES" ]; then TOTAL_IMAGES=0; fi
                    else
                        TOTAL_IMAGES=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
                    fi

                    if [ "$EXISTING_COUNT" -ge "$TOTAL_IMAGES" ] && [ "$TOTAL_IMAGES" -gt 0 ]; then
                        log "    inference result exists (${EXISTING_COUNT}/${TOTAL_IMAGES}), skip"
                    else
                        log "    inferring (arrow/${ARROW_SRC}, base model, no LoRA)... (already have ${EXISTING_COUNT}/${TOTAL_IMAGES})"
                        INFER_CMD="python -m lora.inference.get_triplets_sft_trace_minicpm"
                        INFER_CMD="${INFER_CMD} --output-dir \"${OUTPUT_DIR_ARROW}\""
                        INFER_CMD="${INFER_CMD} --image-dir \"${IMAGE_DIR}\""
                        INFER_CMD="${INFER_CMD} --backend local"
                        INFER_CMD="${INFER_CMD} --base-model-path ${MODEL_PATH}"
                        INFER_CMD="${INFER_CMD} --gpus ${INFERENCE_GPUS}"
                        INFER_CMD="${INFER_CMD} --workers-per-gpu 2"
                        INFER_CMD="${INFER_CMD} --arrow-source ${ARROW_SRC}"
                        if [[ "$dataset" == flowgen_* ]]; then
                            INFER_CMD="${INFER_CMD} --is-flowgen"
                        elif [[ "$dataset" == "bpmn" ]]; then
                            INFER_CMD="${INFER_CMD} --is-bpmn"
                        fi
                        if [ -n "$T_JSON" ]; then
                            INFER_CMD="${INFER_CMD} --test-json ${T_JSON}"
                        fi

                        (cd "<REPO_ROOT>" && eval ${INFER_CMD}) > "${INFER_LOG}" 2>&1
                        INFER_EXIT=$?

                        if [ $INFER_EXIT -ne 0 ]; then
                            log "    ${RED}✗ inference failed (exit=${INFER_EXIT}), see: ${INFER_LOG}${NC}"
                            TASK_IDX=$((TASK_IDX + 1))
                            continue
                        fi
                        log "    inference done"
                    fi

                    if [ ! -f "$EVAL_PY" ]; then
                        log "    ${RED}✗ eval script not found: ${EVAL_PY}${NC}"
                        TASK_IDX=$((TASK_IDX + 1))
                        continue
                    fi

                    log "    scoring..."
                    GT_ROOT_ARG=""
                    if [ -n "${EVAL_GT_ROOT[$dataset]}" ]; then
                        GT_ROOT_ARG="--gt-root ${EVAL_GT_ROOT[$dataset]}"
                    fi
                    EVAL_OUTPUT=$(cd "<REPO_ROOT>" && python "$EVAL_PY" --dataset "$EVAL_DS" --output-dir "$OUTPUT_DIR_ARROW" $GT_ROOT_ARG --image-dir "$IMAGE_DIR" --test-json "${T_JSON:-}" 2>&1)
                    EVAL_EXIT=$?

                    if [ $EVAL_EXIT -ne 0 ]; then
                        log "    ${RED}✗ scoring failed${NC}"
                        log "    ${EVAL_OUTPUT}"
                        TASK_IDX=$((TASK_IDX + 1))
                        continue
                    fi

                    SCORE=$(echo "$EVAL_OUTPUT" | grep -oE 'F1 Score[: ]+[0-9.]+' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
                    if [ -z "$SCORE" ]; then
                        SCORE=$(echo "$EVAL_OUTPUT" | grep -oE '[Ff]1[: ]+[0-9.]+' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
                    fi
                    [ -z "$SCORE" ] && SCORE="N/A"

                    log "    ${GREEN}✓ F1 Score = ${SCORE}${NC}"
                    log ""

                    FINAL_RESULTS["${model_key}|${dataset}|arrow|${ARROW_SRC}"]="${SCORE}"
                    TASK_IDX=$((TASK_IDX + 1))
                done
            fi
        done
    done
done

# ============================================================
# ⑦ summary
# ============================================================
log ""
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  MiniCPM-V4.5 Baseline evaluation results summary${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""

if [[ " ${EVAL_MODES[*]} " == *" triplet "* ]]; then
    log "${CYAN}[Triplet mode (whole-image, get_triplets_sft_e2e_minicpm)]${NC}"
    log "$(printf '  %-18s' 'Model' ${DATASETS[@]})"
    log "$(printf '  %-18s' '-----' $(for d in "${DATASETS[@]}"; do echo '--------'; done))"
    for model_key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
        ROW="  $(printf '%-18s' "$model_key")"
        for dataset in "${DATASETS[@]}"; do
            SCORE="${FINAL_RESULTS[${model_key}|${dataset}|triplet]:-N/A}"
            ROW="${ROW}$(printf '%-18s' "$SCORE")"
        done
        log "$ROW"
    done
    log ""
fi

if [[ " ${EVAL_MODES[*]} " == *" arrow "* ]]; then
    for ARROW_SRC in $ARROW_SOURCES; do
        log "${CYAN}[Arrow mode (per-arrow, get_triplets_sft_trace_minicpm, source=${ARROW_SRC})]${NC}"
        log "$(printf '  %-18s' 'Model' ${DATASETS[@]})"
        log "$(printf '  %-18s' '-----' $(for d in "${DATASETS[@]}"; do echo '--------'; done))"
        for model_key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
            ROW="  $(printf '%-18s' "$model_key")"
            for dataset in "${DATASETS[@]}"; do
                SCORE="${FINAL_RESULTS[${model_key}|${dataset}|arrow|${ARROW_SRC}]:-N/A}"
                ROW="${ROW}$(printf '%-18s' "$SCORE")"
            done
            log "$ROW"
        done
        log ""
    done
fi

# save JSON
SUMMARY_JSON="${EVAL_OUTPUT_DIR}/baseline_summary.json"
python3 << PYEOF
import json

results = {}
FINAL = """$(for key in "${!FINAL_RESULTS[@]}"; do echo "${key}=${FINAL_RESULTS[$key]}"; done)"""

for line in FINAL.strip().split('\n'):
    if '=' not in line:
        continue
    key, val = line.split('=', 1)
    parts = key.split('|')
    model = parts[0]
    dataset = parts[1]
    mode = parts[2]
    arrow_src = parts[3] if len(parts) > 3 else None

    if model not in results:
        results[model] = {}
    if dataset not in results[model]:
        results[model][dataset] = {}

    if arrow_src:
        if mode not in results[model][dataset]:
            results[model][dataset][mode] = {}
        results[model][dataset][mode][arrow_src] = val
    else:
        results[model][dataset][mode] = val

with open("${SUMMARY_JSON}", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Results saved to: ${SUMMARY_JSON}")
PYEOF

log ""
log "${BLUE}  End Time: $(date)${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
