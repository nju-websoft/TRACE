#!/bin/bash

# ============================================================
# eval_all_closed_api.sh
#
# Run two closed-source APIs (GPT-5 / GLM-4.6V) over all 9 datasets to extract
# triplets, then score the predictions with lora/eval/eval_E2E.py.
#
# Env vars (set only the one(s) for the backend you run):
# OPENAI_API_KEY=sk-...
# ZAI_API_KEY=... (or ZHIPU_API_KEY=...; set ZAI_BASE to override the endpoint)
#
# Usage (cd to repository root) :
# cd "<REPO_ROOT>"
# bash scripts/eval_all_closed_api.sh
#
# limit backend / dataset:
# BACKENDS="openai" DATASETS="bpmn fca" bash .../eval_all_closed_api.sh
#
# ============================================================

SCRIPT_CWD="$(pwd)"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

# ============================================================
# ① backend → model mapping
# ============================================================
declare -A MODEL_FOR
MODEL_FOR=(
    ["openai"]="${OPENAI_MODEL:-gpt-5}"
    ["zai"]="${ZAI_MODEL:-glm-4.6v}"
)

BACKENDS_LIST="${BACKENDS:-openai zai}"
ZAI_BASE="${ZAI_BASE:-}"
OPENAI_BASE="${OPENAI_BASE:-}"   # OpenRouter use https://openrouter.ai/api/v1

# ============================================================
# ② 9 dataset ( vs eval_baseline_no_sft.sh consistent)
# ============================================================
DATASETS_DEFAULT="fca cbd fcb flowlearn flowvqa bpmn flowgen_easy flowgen_medium flowgen_hard"
DATASETS_LIST="${DATASETS:-$DATASETS_DEFAULT}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
MAX_IMAGES="${MAX_IMAGES:-0}"   # 0 = run all; >0 = each dataset before N images

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
    ["fcb"]=""
    ["flowlearn"]="<DATA_ROOT>/flowlearn/test.json"
    ["flowvqa"]="<DATA_ROOT>/flowvqa/Data/test_full.json"
    ["bpmn"]=""
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
# based on GT real triplet count: max_triplet × single triplet tokens × 1.5 buffer, then +500 buffer
TEST_MAX_TOKENS=(
    ["fca"]="1024"            # max 13 triplets
    ["cbd"]="1024"            # max 16
    ["fcb"]="1024"            # max 11
    ["flowlearn"]="1024"      # max 13
    ["flowgen_easy"]="1024"   # max 13
    ["flowvqa"]="2048"        # max 41
    ["flowgen_medium"]="1536" # max 45
    ["bpmn"]="2560"           # max 80
    ["flowgen_hard"]="2560"   # max 74
)

# evaluation script (unified entry, eval_E2E.py + --dataset)
EVAL_PY_TRIPLET="<REPO_ROOT>/lora/eval/eval_E2E.py"
declare -A EVAL_DATASET
EVAL_DATASET=(
    ["fca"]="fca" ["cbd"]="cbd" ["fcb"]="fcb"
    ["flowlearn"]="flowlearn" ["flowvqa"]="flowvqa" ["bpmn"]="bpmn"
    ["flowgen_easy"]="flowgen" ["flowgen_medium"]="flowgen" ["flowgen_hard"]="flowgen"
)

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-<OUTPUT_ROOT>/output_closed_api}"
PY_BIN="${PY_BIN:-<CONDA_ROOT>/envs/<CONDA_ENV>/bin/python}"

# ============================================================
# ③ logging
# ============================================================
LOG_DIR="${SCRIPT_CWD}/training_logs"
mkdir -p "$LOG_DIR" "$EVAL_OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/closed_api_all_${TIMESTAMP}.log"

log() { echo -e "$@" | tee -a "$MAIN_LOG"; }

log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  closed-source API evaluation (9 datasets, triplet mode)${NC}"
log "${BLUE}  Backends:${NC}"
for b in $BACKENDS_LIST; do
    log "${BLUE}    ${b} → ${MODEL_FOR[$b]}${NC}"
done
log "${BLUE}  Datasets: ${DATASETS_LIST}${NC}"
log "${BLUE}  Output dir: ${EVAL_OUTPUT_DIR}${NC}"
log "${BLUE}  Start: $(date)${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"

check_key () {
    case "$1" in
        openai) [ -z "$OPENAI_API_KEY" ] && [ -z "$OPENROUTER_API_KEY" ] && return 1 ;;
        zai)    [ -z "$ZAI_API_KEY" ] && [ -z "$ZHIPU_API_KEY" ] && return 1 ;;
    esac
    return 0
}

extract_f1() {
    local output_json="$1"
    $PY_BIN -c "
import os, json, glob
result_dir = os.path.dirname('$output_json')
stem = os.path.splitext(os.path.basename('$output_json'))[0]
cands = glob.glob(os.path.join(result_dir, f'evaluation_results*{stem}*.json'))
if cands:
    d = json.load(open(cands[0]))
    if 'f1_score' in d:
        print(f'{d[\"f1_score\"]:.4f}')
" 2>/dev/null
}

declare -A FINAL_F1
TOTAL_TASKS=0
for backend in $BACKENDS_LIST; do
    for ds in $DATASETS_LIST; do
        TOTAL_TASKS=$((TOTAL_TASKS + 1))
    done
done
log "total tasks: ${TOTAL_TASKS}"
TASK_IDX=0

# ============================================================
# ④ main loop
# ============================================================
for backend in $BACKENDS_LIST; do
    model="${MODEL_FOR[$backend]}"
    if [ -z "$model" ]; then
        log "${YELLOW}  ⊘ Skip backend=${backend}: unknown${NC}"
        continue
    fi
    if ! check_key "$backend"; then
        log "${YELLOW}  ⊘ Skip ${backend} (${model}): API key not set${NC}"
        continue
    fi

    safe_model="$(echo "${model}" | tr '/.' '__')"

    for ds in $DATASETS_LIST; do
        TASK_IDX=$((TASK_IDX + 1))
        IMAGE_DIR="${TEST_IMAGE_DIR[$ds]}"
        T_JSON="${TEST_JSON[$ds]}"
        IMG_EXT="${TEST_IMAGE_EXT[$ds]}"
        MAX_TOK="${TEST_MAX_TOKENS[$ds]}"
        EVAL_PY="$EVAL_PY_TRIPLET"
        EVAL_DS="${EVAL_DATASET[$ds]}"

        if [ -z "$IMAGE_DIR" ] || [ ! -d "$IMAGE_DIR" ]; then
            log "${RED}  [${TASK_IDX}/${TOTAL_TASKS}] ✗ ${backend}/${ds}: image_dir not found${NC}"
            FINAL_F1["${backend}|${ds}"]="N/A"
            continue
        fi
        if [ ! -f "$EVAL_PY" ]; then
            log "${RED}  [${TASK_IDX}/${TOTAL_TASKS}] ✗ ${backend}/${ds}: eval script not found${NC}"
            FINAL_F1["${backend}|${ds}"]="N/A"
            continue
        fi

        OUTPUT_JSON="${EVAL_OUTPUT_DIR}/${backend}_${safe_model}_${ds}_triplet.json"
        INFER_LOG="${LOG_DIR}/infer_closed_${backend}_${safe_model}_${ds}_${TIMESTAMP}.log"

        log ""
        log "${GREEN}  ▶ [${TASK_IDX}/${TOTAL_TASKS}] ${backend} / ${ds} (model=${model})${NC}"
        log "    image_dir: ${IMAGE_DIR}"
        log "    output:    ${OUTPUT_JSON}"

        # ── inference ──
        INFER_CMD="$PY_BIN -m lora.inference.get_triplets_closed_api"
        INFER_CMD+=" --backend ${backend}"
        INFER_CMD+=" --model ${model}"
        INFER_CMD+=" --image-dir \"${IMAGE_DIR}\""
        INFER_CMD+=" --output-json \"${OUTPUT_JSON}\""
        INFER_CMD+=" --image-ext ${IMG_EXT}"
        INFER_CMD+=" --max-new-tokens ${MAX_TOK}"
        INFER_CMD+=" --max-concurrent ${MAX_CONCURRENT}"
        if [ "$MAX_IMAGES" -gt 0 ] 2>/dev/null; then
            INFER_CMD+=" --max-images ${MAX_IMAGES}"
        fi
        if [ -n "$T_JSON" ] && [ -f "$T_JSON" ]; then
            INFER_CMD+=" --test-json \"${T_JSON}\""
        fi
        if [ "$backend" = "zai" ] && [ -n "$ZAI_BASE" ]; then
            INFER_CMD+=" --api-base ${ZAI_BASE}"
        fi
        if [ "$backend" = "openai" ] && [ -n "$OPENAI_BASE" ]; then
            INFER_CMD+=" --api-base ${OPENAI_BASE}"
        fi

        log "    inferring... (log: ${INFER_LOG})"
        (cd "<REPO_ROOT>" && eval ${INFER_CMD}) > "${INFER_LOG}" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
            log "    ${RED}✗ inference failed (exit=${rc}), see ${INFER_LOG}${NC}"
            FINAL_F1["${backend}|${ds}"]="ERR"
            continue
        fi
        log "    inference done"

        # ── scoring ──
        # eval_E2E.py to flowgen auto from prediction identify from filename split (easy/medium/hard), no need --gt-root
        log "    scoring..."
        EVAL_CMD=("$PY_BIN" "$EVAL_PY" --dataset "$EVAL_DS" --prediction-file "$OUTPUT_JSON" --image-dir "$IMAGE_DIR" --test-json "${T_JSON:-}")
        EVAL_OUTPUT=$(cd "<REPO_ROOT>" && "${EVAL_CMD[@]}" 2>&1)
        rc=$?
        if [ $rc -ne 0 ]; then
            log "    ${RED}✗ scoring failed${NC}"
            log "$EVAL_OUTPUT"
            FINAL_F1["${backend}|${ds}"]="N/A"
            continue
        fi

        SCORE=$(extract_f1 "$OUTPUT_JSON")
        [ -z "$SCORE" ] && SCORE=$(echo "$EVAL_OUTPUT" | grep -oE 'F1[: ]+[0-9.]+' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        [ -z "$SCORE" ] && SCORE="N/A"
        log "    ${GREEN}✓ F1 = ${SCORE}${NC}"
        FINAL_F1["${backend}|${ds}"]="${SCORE}"
    done
done

# ============================================================
# ⑤ summary
# ============================================================
log ""
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  summary - F1${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"

# header row
header="$(printf '  %-22s' 'Backend / Model')"
for ds in $DATASETS_LIST; do
    header+="$(printf '%-12s' "$ds")"
done
log "$header"
sep="  $(printf '%22s' '' | tr ' ' '-')"
for ds in $DATASETS_LIST; do
    sep+="$(printf '%12s' '' | tr ' ' '-')"
done
log "$sep"

for backend in $BACKENDS_LIST; do
    model="${MODEL_FOR[$backend]}"
    row="$(printf '  %-22s' "${backend}/${model}")"
    for ds in $DATASETS_LIST; do
        score="${FINAL_F1[${backend}|${ds}]:-not run}"
        row+="$(printf '%-12s' "$score")"
    done
    log "$row"
done

SUMMARY_JSON="${EVAL_OUTPUT_DIR}/closed_api_summary_${TIMESTAMP}.json"
$PY_BIN - <<PYEOF
import json
res = {}
raw = """$(for k in "${!FINAL_F1[@]}"; do echo "${k}=${FINAL_F1[$k]}"; done)"""
for line in raw.strip().split("\n"):
    if "=" not in line: continue
    k, v = line.split("=", 1)
    backend, ds = k.split("|", 1)
    res.setdefault(backend, {})[ds] = v
json.dump(res, open("${SUMMARY_JSON}", "w"), indent=2, ensure_ascii=False)
print("Summary saved:", "${SUMMARY_JSON}")
PYEOF

log ""
log "${BLUE}  End: $(date)${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
