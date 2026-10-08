#!/bin/bash

# ============================================================
# train_leave_one_out_E2E.sh
#
# Triplet generalization (Leave-One-Out) training script
#
# approach: for each dataset L (target):
# - merge *the rest 8 * dataset triplet training set → random shuffle → loo_train_"<L>.json"
# - skip val (epoch=1, early stopping patience=1 meaningless, skip eval whenbetween)
# - batch_size = the rest 8 dataset (in train_all_triplet_grid_search.sh
# in) best BS *median* (dynamically compute, no need to fill manually)
# strategy: per_device = round(grid_global_bs / history_NUM_DEVICES),
# history ND from training_logs/ 'Num Devices: N' reclaim,
# new_global_bs = mode(per_device 8 ) × NUM_DEVICES
# - num_epochs = 1, save once at end checkpoint
# training done, after use eval_loo_triplet.sh in L test on the test set, get generalization performance.
#
# Usage:
# cd MLLMs_SFT/Qwen-VL-Series-Finetune
# bash scripts/train_leave_one_out_E2E.sh
#
# env var (optional):
# LOO_TARGETS override default leave-out target list (space-separated)
# FORCE_REBUILD =1 force rebuild merged when set JSON
# ONLY_BUILD =1 only generate merged when set JSON, nottraining (for dry-run / debug)
# ============================================================

# SCRIPT_CWD always points to MLLMs_SFT/Qwen-VL-Series-Finetune (script's dir parent),
# so no matter where cwd launch can find it output/ and training_logs/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_CWD="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SCRIPT_CWD"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================
# ① Model config ( vs train_all_triplet_grid_search.sh keep consistent)
# ============================================================
declare -A MODELS
MODELS=(
    ["qwen3_vl_4b"]="<MODEL_ROOT>/Qwen3-VL-4B-Instruct"
)


# ============================================================
# ② all dataset (9 )
# ============================================================
ALL_DATASETS=("fca" "fcb" "flowlearn" "flowgen_hard" "flowvqa" "bpmn" "flowgen_easy" "flowgen_medium" "cbd")

if [ -n "$LOO_TARGETS" ]; then
    IFS=' ' read -r -a _LOO_TARGETS <<< "$LOO_TARGETS"
else
    _LOO_TARGETS=("${ALL_DATASETS[@]}")
fi

# skip the training stage for a target (data merge/BS compute as usual, only skip the actual training)
SKIP_TRAIN_TARGETS=()
is_skip_train_target() {
    local t="$1"
    for s in "${SKIP_TRAIN_TARGETS[@]}"; do
        [ "$s" = "$t" ] && return 0
    done
    return 1
}

# ============================================================
# ③ Path config
# ============================================================
TRIPLET_DATA_DIR="<DATA_ROOT>/data_4_triplet_training"
LOO_DATA_DIR="${TRIPLET_DATA_DIR}/leave_one_out"
OUTPUT_BASE_DIR="${SCRIPT_CWD}/output"
TRAIN_SCRIPT="${SCRIPT_CWD}/scripts/train_early_stop.sh"   # not passed --eval_data_path auto falls back to no early stoppingtraining

# ============================================================
# ④ trainingargs
# ============================================================
NUM_EPOCHS=1
NUM_DEVICES=8
CUDA_DEVICES="0,1,2,3,4,5,6,7"
SHUFFLE_SEED=42
SAVE_INTERVAL=1.0   # 1   epoch save once at the end

# history grid-search each run use device count not fixed, from training_logs/ in
# 'Num Devices: N' auto-reclaim (each run one line) .
LOG_DIR_FOR_HIST="${SCRIPT_CWD}/training_logs"

# ============================================================
# ⑤ logging
# ============================================================
LOG_DIR="${SCRIPT_CWD}/training_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/loo_triplet_${TIMESTAMP}.log"
log() { echo -e "$@" | tee -a "$MAIN_LOG"; }

log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  Leave-One-Out generalizationtraining (Triplet, no val, epoch=${NUM_EPOCHS})${NC}"
log "${BLUE}  Models:    ${!MODELS[*]}${NC}"
log "${BLUE}  Targets:   ${_LOO_TARGETS[*]}${NC}"
log "${BLUE}  Epochs:    ${NUM_EPOCHS}${NC}"
log "${BLUE}  Time:      $(date)${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""

mkdir -p "$LOO_DATA_DIR"

# ============================================================
# ⑥ Stage 0: merge + shuffle train / val JSON (each target one-to-)
# ============================================================
log "${CYAN}[Stage 0] generate leave-one-out data (merge + shuffle, only train)${NC}"
log ""

python3 - "$LOO_DATA_DIR" "$TRIPLET_DATA_DIR" "$SHUFFLE_SEED" "${ALL_DATASETS[*]}" "${_LOO_TARGETS[*]}" "${FORCE_REBUILD:-0}" << 'PYEOF' 2>&1 | while IFS= read -r line; do log "  $line"; done
import json, os, random, sys

loo_dir, triplet_dir, seed_str, all_str, targets_str, force_str = sys.argv[1:7]
seed = int(seed_str)
all_ds = all_str.split()
targets = targets_str.split()
force = force_str == "1"

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get_train_path(ds):
    return os.path.join(triplet_dir, f"{ds}_triplet_training_all.json")

# full pre-check
for ds in all_ds:
    tp = get_train_path(ds)
    if not os.path.exists(tp):
        print(f"[FATAL] missingtraining JSON: {tp}")
        sys.exit(1)

for target in targets:
    others = [d for d in all_ds if d != target]
    out_train = os.path.join(loo_dir, f"loo_train_{target}.json")

    if not force and os.path.exists(out_train):
        n_t = len(load_json(out_train))
        print(f"[skip] target={target}: exists train={n_t}")
        continue

    train_items = []
    src_counts = {}
    for ds in others:
        items = load_json(get_train_path(ds))
        for it in items:
            if isinstance(it, dict) and "_src" not in it:
                it["_src"] = ds
        train_items.extend(items)
        src_counts[ds] = len(items)

    rng = random.Random(seed)
    rng.shuffle(train_items)

    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_items, f, ensure_ascii=False)

    breakdown = ", ".join(f"{k}={v}" for k, v in src_counts.items())
    print(f"[built] target={target}: train={len(train_items)}")
    print(f"        source: {breakdown}")
PYEOF

if [ "${ONLY_BUILD:-0}" == "1" ]; then
    log ""
    log "${YELLOW}ONLY_BUILD=1, skip trainingStage, end. ${NC}"
    exit 0
fi

# ============================================================
# ⑦ Stage 1: scan existing grid-search results, for each target compute BS median
# ============================================================
log ""
log "${CYAN}[Stage 1] compute each  target   BS median${NC}"
log "  strategy: per_device = round(grid_global_bs / history_NUM_DEVICES), history ND from training_logs/ reclaim"
log "        new_global_bs = mode(per_device 8  ) × NUM_DEVICES=${NUM_DEVICES}"
log ""

MODELS_KEYS_STR=$(echo "${!MODELS[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')

# output format: model_key|target|bs_global_new|reason
BS_MODES=$(python3 - "$OUTPUT_BASE_DIR" "$LOG_DIR_FOR_HIST" "$MODELS_KEYS_STR" "${_LOO_TARGETS[*]}" "${ALL_DATASETS[*]}" "$NUM_DEVICES" << 'PYEOF'
import glob, json, os, re, sys
from collections import Counter

output_base, log_dir, models_str, targets_str, all_str, cur_nd_str = sys.argv[1:7]
models = models_str.split()
targets = targets_str.split()
all_ds = all_str.split()
cur_nd = int(cur_nd_str)

def best_bs_for(model_key, ds):
    """scan output/{model}_triplet_{ds}_bs*/, pick min(eval_loss) corresponding  *global* BS. """
    pattern = os.path.join(output_base, f"{model_key}_triplet_{ds}_bs*")
    runs = sorted(glob.glob(pattern))
    best_loss = float("inf")
    best_bs = None
    for run in runs:
        m = re.search(r"_bs(\d+)$", run)
        if not m:
            continue
        bs = int(m.group(1))
        state_file = os.path.join(run, "trainer_state.json")
        if os.path.isfile(state_file):
            with open(state_file) as f:
                state = json.load(f)
        else:
            ckpt_states = glob.glob(os.path.join(run, "checkpoint-*", "trainer_state.json"))
            if not ckpt_states:
                continue
            ckpt_states.sort(key=lambda x: int(x.split("checkpoint-")[1].split("/")[0]))
            with open(ckpt_states[-1]) as f:
                state = json.load(f)
        evals = [e for e in state.get("log_history", []) if "eval_loss" in e]
        if not evals:
            continue
        run_best = min(e["eval_loss"] for e in evals)
        if run_best < best_loss:
            best_loss = run_best
            best_bs = bs
    return best_bs

def find_hist_num_devices(model_key, ds, bs):
    """from training_logs/{model}_triplet_{ds}_bs{bs}_*.log in grep 'Num Devices: N'. 
    multiple  log take latest timestamp that . """
    pat = os.path.join(log_dir, f"{model_key}_triplet_{ds}_bs{bs}_*.log")
    logs = sorted(glob.glob(pat), reverse=True)
    for lp in logs:
        try:
            with open(lp, "r", errors="ignore") as f:
                txt = f.read()
        except OSError:
            continue
        m = re.search(r"Num Devices:\s*(\d+)", txt)
        if m:
            return int(m.group(1))
    return None

for model_key in sorted(models):
    for target in targets:
        others = [d for d in all_ds if d != target]
        per_dev_list = []
        detail_items = []
        missing = []
        for ds in others:
            gbs = best_bs_for(model_key, ds)
            if gbs is None:
                missing.append(f"{ds}(no_run)")
                continue
            hist_nd = find_hist_num_devices(model_key, ds, gbs)
            if hist_nd is None:
                missing.append(f"{ds}(no_log)")
                continue
            pd = max(1, int(round(gbs / hist_nd)))
            per_dev_list.append(pd)
            detail_items.append(f"{ds}:gbs={gbs},nd={hist_nd},pd={pd}")
        if not per_dev_list:
            print(f"{model_key}|{target}|0|NO_DATA missing={','.join(missing) or '-'}")
            continue
        c = Counter(per_dev_list)
        max_count = max(c.values())
        modes = sorted([k for k, v in c.items() if v == max_count])
        pd_mode = modes[0]   # on tie take smaller  (more conservative) 
        new_gbs = pd_mode * cur_nd
        reason = f"pd_list={per_dev_list} pd_mode={pd_mode} | {' '.join(detail_items)}"
        if missing:
            reason += f" | missing={','.join(missing)}"
        print(f"{model_key}|{target}|{new_gbs}|{reason}")
PYEOF
)

log "already computed  BS median (global = per_device_mode × ${NUM_DEVICES}):"
log "$(printf '  %-16s %-16s %-10s' 'Model' 'Target' 'NewGlobalBS')"
log "$(printf '  %-16s %-16s %-10s' '-----' '------' '-----------')"
while IFS='|' read -r mk tgt bs reason; do
    [ -z "$mk" ] && continue
    log "$(printf '  %-16s %-16s %-10s' "$mk" "$tgt" "$bs")"
    log "      ${reason}"
done <<< "$BS_MODES"
log ""

# ============================================================
# ⑧ Stage 2: training (no val, 1 epoch)
# ============================================================
log "${CYAN}[Stage 2] training leave-one-out (Triplet) (epoch=${NUM_EPOCHS}, no val) ${NC}"
log ""

TOTAL=0
DONE=0
SKIPPED=0
FAILED=0
TRAINED_TARGETS=()   # trainingsuccess / exists ckpt   target, for auto-eval
while IFS='|' read -r mk tgt bs _; do
    [ -z "$mk" ] && continue
    [ "$bs" = "0" ] && continue
    is_skip_train_target "$tgt" && continue
    TOTAL=$((TOTAL + 1))
done <<< "$BS_MODES"

log "total training tasks: ${TOTAL}    (skip target: ${SKIP_TRAIN_TARGETS[*]:-no})"
log ""

while IFS='|' read -r model_key target bs_mode reason; do
    [ -z "$model_key" ] && continue
    if is_skip_train_target "$target"; then
        log "${YELLOW}  ⊘ ${model_key} / loo_${target}: in SKIP_TRAIN_TARGETS in, skip training${NC}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    if [ "$bs_mode" = "0" ]; then
        log "${YELLOW}  ⊘ ${model_key} / loo_${target}: none available BS data, skip (${reason})${NC}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    MODEL_PATH="${MODELS[$model_key]}"
    if [ -z "$MODEL_PATH" ]; then
        log "${RED}  ✗ ${model_key}: not in MODELS defined in, skip${NC}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    TRAIN_JSON="${LOO_DATA_DIR}/loo_train_${target}.json"
    if [ ! -f "$TRAIN_JSON" ]; then
        log "${RED}  ✗ ${target}: merge JSON missing, skip${NC}"
        FAILED=$((FAILED + 1))
        continue
    fi
    N_TRAIN=$(python3 -c "import json; print(len(json.load(open('$TRAIN_JSON'))))")

    RUN_NAME="${model_key}_triplet_loo_${target}_bs${bs_mode}_e${NUM_EPOCHS}"
    RUN_OUT="${OUTPUT_BASE_DIR}/${RUN_NAME}"
    TASK_LOG="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"

    log "${BLUE}────────────────────────────────────────────────────────────${NC}"
    log "${GREEN}  ▶ ${model_key} / leave-out=${target}${NC}"
    log "    BS:        ${bs_mode}    (${reason})"
    log "    train:     ${TRAIN_JSON}  (n=${N_TRAIN})"
    log "    out:       ${RUN_OUT}"
    log "    task log:  ${TASK_LOG}"

    if [ -d "$RUN_OUT" ]; then
        CKPT_COUNT=$(find "$RUN_OUT" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)
        if [ "$CKPT_COUNT" -gt 0 ]; then
            log "    ${CYAN}already have ${CKPT_COUNT}   checkpoint, skip${NC}"
            SKIPPED=$((SKIPPED + 1))
            TRAINED_TARGETS+=("$target")
            continue
        fi
    fi

    bash "$TRAIN_SCRIPT" \
        --model_name "$MODEL_PATH" \
        --cuda_devices "$CUDA_DEVICES" \
        --data_path "$TRAIN_JSON" \
        --output_dir "$RUN_OUT" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$bs_mode" \
        --num_devices "$NUM_DEVICES" \
        --save_interval "$SAVE_INTERVAL" \

        > "$TASK_LOG" 2>&1
    EC=$?

    if [ $EC -eq 0 ]; then
        log "    ${GREEN}✓ success${NC}"
        DONE=$((DONE + 1))
        TRAINED_TARGETS+=("$target")
    else
        log "    ${RED}✗ failed (exit=${EC})${NC}"
        tail -8 "$TASK_LOG" 2>/dev/null | while read line; do log "      $line"; done
        FAILED=$((FAILED + 1))
    fi
done <<< "$BS_MODES"

log ""
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  training done: success=${DONE}  skip=${SKIPPED}  failed=${FAILED}  /  total=${TOTAL}${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""


# ============================================================
# ⑨ Stage 3: auto-eval (SKIP_EVAL=1 can skip)
# ============================================================
if [ "${SKIP_EVAL:-0}" = "1" ]; then
    log "${YELLOW}SKIP_EVAL=1, skipauto-eval${NC}"
    exit 0
fi

# deduptrainingsuccess targets
declare -A _SEEN
EVAL_TARGETS_LIST=""
for t in "${TRAINED_TARGETS[@]}"; do
    if [ -z "${_SEEN[$t]:-}" ]; then
        _SEEN["$t"]=1
        EVAL_TARGETS_LIST="${EVAL_TARGETS_LIST}${t} "
    fi
done
EVAL_TARGETS_LIST="${EVAL_TARGETS_LIST% }"

if [ -z "$EVAL_TARGETS_LIST" ]; then
    log "${YELLOW}nothing to eval  target, skip${NC}"
    exit 0
fi

read -r -a _TARGETS <<< "$EVAL_TARGETS_LIST"

# ── eval config ──
EVAL_OUTPUT_DIR="<OUTPUT_ROOT>/output_lora_triplet_loo"
INFERENCE_GPUS="${INFERENCE_GPUS:-$CUDA_DEVICES}"

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
    ["flowlearn"]="<DATA_ROOT>/flowlearn/test.json"
    ["flowvqa"]="<DATA_ROOT>/flowvqa/Data/test_full.json"
)
declare -A IMAGE_EXT
IMAGE_EXT=(
    ["fca"]=".png" ["cbd"]=".png" ["fcb"]=".png"
    ["flowlearn"]=".jpeg" ["flowvqa"]=".png" ["bpmn"]=".png"
    ["flowgen_easy"]=".png" ["flowgen_medium"]=".png" ["flowgen_hard"]=".png"
)
declare -A MAX_NEW_TOKENS
MAX_NEW_TOKENS=(
    ["fca"]="1024" ["cbd"]="2048" ["fcb"]="2048" ["flowlearn"]="2048"
    ["flowvqa"]="4096" ["bpmn"]="4096"
    ["flowgen_easy"]="4096" ["flowgen_medium"]="4096" ["flowgen_hard"]="4096"
)

EVAL_PY_BIN="<REPO_ROOT>/lora/eval/eval_E2E.py"
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

log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  [Stage 3] LOO auto-eval${NC}"
log "${BLUE}  Models:  ${!MODELS[*]}${NC}"
log "${BLUE}  Targets: ${EVAL_TARGETS_LIST}${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""

# scan LOO training dir, find step max checkpoint
LOO_RUNS=$(python3 - "$OUTPUT_BASE_DIR" "$(echo "${!MODELS[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')" "${_TARGETS[*]}" << 'PYEOF'
import glob, os, re, sys
output_base, models_str, targets_str = sys.argv[1:4]
for mk in models_str.split():
    for tgt in targets_str.split():
        pat = os.path.join(output_base, f"{mk}_triplet_loo_{tgt}_bs*_e*")
        for run in sorted(glob.glob(pat)):
            m = re.search(r"_bs(\d+)_e(\d+)$", run)
            if not m: continue
            bs = int(m.group(1))
            ckpts = []
            for c in glob.glob(os.path.join(run, "checkpoint-*")):
                mc = re.search(r"checkpoint-(\d+)$", c)
                if mc and os.path.isdir(c):
                    ckpts.append((int(mc.group(1)), c))
            if not ckpts: continue
            ckpts.sort()
            step, ckpt_dir = ckpts[-1]
            print(f"{mk}|{tgt}|{bs}|{ckpt_dir}|{step}")
PYEOF
)

if [ -z "$LOO_RUNS" ]; then
    log "${RED}none found LOO training result, skipeval${NC}"
    exit 0
fi

mkdir -p "$EVAL_OUTPUT_DIR"
declare -A FINAL_RESULTS
TASK_COUNT=0
DONE_COUNT=0
FAIL_COUNT=0
while IFS='|' read -r _ _ _ _ _; do
    [ -n "$_" ] && TASK_COUNT=$((TASK_COUNT + 1))
done <<< "$LOO_RUNS"

while IFS='|' read -r model_key target bs ckpt_dir step; do
    [ -z "$model_key" ] && continue

    MODEL_PATH="${MODELS[$model_key]}"
    EVAL_DS="${EVAL_DATASET[$target]}"
    IMAGE_DIR="${TEST_IMAGE_DIR[$target]}"
    T_JSON="${TEST_JSON[$target]:-}"
    EXT="${IMAGE_EXT[$target]}"
    MAX_TOK="${MAX_NEW_TOKENS[$target]:-4096}"

    if [ -z "$MODEL_PATH" ] || [ -z "$IMAGE_DIR" ] || [ -z "$EVAL_DS" ]; then
        log "${YELLOW}  ⊘ ${model_key}/loo_${target}: incomplete config, skip${NC}"
        continue
    fi

    RESULT_DIR="${EVAL_OUTPUT_DIR}/${model_key}_triplet_loo_${target}_bs${bs}_step${step}"
    OUTPUT_JSON="${RESULT_DIR}/predictions.json"
    mkdir -p "$RESULT_DIR"

    log "${GREEN}  ▶ [${DONE_COUNT}/${TASK_COUNT}] ${model_key} / leave-out=${target}${NC}"
    log "    BS=${bs}  ckpt=${ckpt_dir}"

    # inference
    INFER_CMD="python -m lora.inference.get_triplets_sft_e2e"
    INFER_CMD="${INFER_CMD} --base-model-path ${MODEL_PATH}"
    INFER_CMD="${INFER_CMD} --lora-path ${ckpt_dir}"
    INFER_CMD="${INFER_CMD} --image-dir \"${IMAGE_DIR}\""
    INFER_CMD="${INFER_CMD} --output-json \"${OUTPUT_JSON}\""
    INFER_CMD="${INFER_CMD} --backend local"
    INFER_CMD="${INFER_CMD} --gpus ${INFERENCE_GPUS}"
    INFER_CMD="${INFER_CMD} --workers-per-gpu 2"
    INFER_CMD="${INFER_CMD} --image-ext ${EXT}"
    INFER_CMD="${INFER_CMD} --max-new-tokens ${MAX_TOK}"
    [ -n "$T_JSON" ] && INFER_CMD="${INFER_CMD} --test-json ${T_JSON}"
    INFER_LOG="${LOG_DIR}/loo_E2E_infer_${model_key}_${target}_bs${bs}_${TIMESTAMP}.log"

    if [ -f "$OUTPUT_JSON" ] && [ "${FORCE_RERUN:-0}" != "1" ]; then
        log "    predictionexists, skip inference"
    else
        log "    inferring..."
        (cd "<REPO_ROOT>" && eval ${INFER_CMD}) > "${INFER_LOG}" 2>&1
        if [ $? -ne 0 ]; then
            log "    ${RED}✗ inference failed, see ${INFER_LOG}${NC}"
            tail -10 "$INFER_LOG" 2>/dev/null | while read l; do log "      $l"; done
            FAIL_COUNT=$((FAIL_COUNT + 1))
            continue
        fi
    fi

    if [ ! -f "$OUTPUT_JSON" ]; then
        log "    ${RED}✗ prediction JSON not generated${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # scoring
    EVAL_EXTRA=""
    [ -n "${EVAL_GT_ROOT[$target]}" ] && EVAL_EXTRA="--gt-root ${EVAL_GT_ROOT[$target]}"
    EVAL_OUTPUT=$(cd "<REPO_ROOT>" && python "$EVAL_PY_BIN" --dataset "$EVAL_DS" --prediction-file "$OUTPUT_JSON" $EVAL_EXTRA --image-dir "$IMAGE_DIR" --test-json "${T_JSON:-}" 2>&1)
    if [ $? -ne 0 ]; then
        log "    ${RED}✗ scoring failed${NC}"
        echo "$EVAL_OUTPUT" | head -10 | while read l; do log "      $l"; done
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    SCORE=$(echo "$EVAL_OUTPUT" | grep -oE 'F1 Score:[[:space:]]*[0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
    [ -z "$SCORE" ] && SCORE="N/A"
    log "    ${GREEN}✓ F1 = ${SCORE}${NC}"
    FINAL_RESULTS["${model_key}|${target}"]="${bs}|${step}|${SCORE}"
    DONE_COUNT=$((DONE_COUNT + 1))

done <<< "$LOO_RUNS"

# summary
log ""
log "${BLUE}════════ LOO E2E eval results ════════${NC}"
log "$(printf '  %-16s %-16s %-6s %-8s %-10s' 'Model' 'LeaveOut' 'BS' 'Step' 'F1')"
for key in $(echo "${!FINAL_RESULTS[@]}" | tr ' ' '\n' | sort); do
    IFS='|' read -r mk tgt <<< "$key"
    IFS='|' read -r bs step score <<< "${FINAL_RESULTS[$key]}"
    log "$(printf '  %-16s %-16s %-6s %-8s %-10s' "$mk" "$tgt" "$bs" "$step" "$score")"
done

SUMMARY_JSON="${EVAL_OUTPUT_DIR}/loo_E2E_eval_${TIMESTAMP}.json"
python3 -c "
import json
res = {}
raw = '''$(for k in "${!FINAL_RESULTS[@]}"; do echo "${k}=${FINAL_RESULTS[$k]}"; done)'''
for line in raw.strip().split('\n'):
    if not line.strip(): continue
    k, v = line.split('=', 1)
    mk, tgt = k.split('|')
    bs, step, score = v.split('|')
    res[f'{mk}/loo_{tgt}'] = {'batch_size': int(bs), 'step': int(step), 'test_f1': score}
json.dump(res, open('$SUMMARY_JSON', 'w'), indent=2, ensure_ascii=False)
print('results saved:', '$SUMMARY_JSON')
" 2>&1 | while read l; do log "  $l"; done

log ""
log "  eval: done=${DONE_COUNT}  failed=${FAIL_COUNT}  total=${TASK_COUNT}"
[ "$FAIL_COUNT" -gt 0 ] && exit 1 || exit 0
