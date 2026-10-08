#!/bin/bash

# ============================================================
# train_leave_one_out_TRACE.sh
#
# Arrow generalization (Leave-One-Out) training script
#
# approach: for each dataset L (target):
# - merge *the rest 8 * dataset arrow training set → random shuffle → loo_train_"<L>.json"
# - skip val (user requested "first onlytraining1 epochcheck results")
# - batch_size = the rest 8 dataset (in train_all_arrow_grid_search.sh
# in) best BS *median* (dynamically compute, no need to fill manually)
# - num_epochs = 1
# training done, after use cross_test_arrow.sh in L test on the test set, get generalization performance.
#
# Usage:
# cd MLLMs_SFT/Qwen-VL-Series-Finetune
# bash scripts/train_leave_one_out_TRACE.sh
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
# ① Model config ( vs train_all_arrow_grid_search.sh keep consistent)
# ============================================================
declare -A MODELS
MODELS=(
    ["qwen3_vl_4b"]="<MODEL_ROOT>/Qwen3-VL-4B-Instruct"
)


# ============================================================
# ② all dataset (9 )
# ============================================================
ALL_DATASETS=("fca" "fcb" "flowlearn" "flowgen_hard" "flowvqa" "bpmn" "flowgen_easy" "flowgen_medium"  "cbd" )

if [ -n "$LOO_TARGETS" ]; then
    IFS=' ' read -r -a _LOO_TARGETS <<< "$LOO_TARGETS"
else
    _LOO_TARGETS=("${ALL_DATASETS[@]}")
fi

# skip the training stage for a target (data merge/BS compute as usual, only skip the actual training)
SKIP_TRAIN_TARGETS=("fca" "fcb" "flowlearn" "flowgen_hard" "flowvqa")
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
ARROW_DATA_DIR="<DATA_ROOT>/data_4_training"
LOO_DATA_DIR="${ARROW_DATA_DIR}/leave_one_out"
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
MAIN_LOG="${LOG_DIR}/loo_arrow_${TIMESTAMP}.log"
log() { echo -e "$@" | tee -a "$MAIN_LOG"; }

log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  Leave-One-Out generalizationtraining (Arrow, no val, epoch=1)${NC}"
log "${BLUE}  Models:    ${!MODELS[*]}${NC}"
log "${BLUE}  Targets:   ${_LOO_TARGETS[*]}${NC}"
log "${BLUE}  Epochs:    ${NUM_EPOCHS}${NC}"
log "${BLUE}  Time:      $(date)${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""

mkdir -p "$LOO_DATA_DIR"

# ============================================================
# ⑥ Stage 0: merge + shuffle train JSON (each target a copy)
# ============================================================
log "${CYAN}[Stage 0] generate leave-one-out data (merge + shuffle, only train)${NC}"
log ""

python3 - "$LOO_DATA_DIR" "$ARROW_DATA_DIR" "$SHUFFLE_SEED" "${ALL_DATASETS[*]}" "${_LOO_TARGETS[*]}" "${FORCE_REBUILD:-0}" << 'PYEOF' 2>&1 | while IFS= read -r line; do log "  $line"; done
import json, os, random, sys

loo_dir, arrow_dir, seed_str, all_str, targets_str, force_str = sys.argv[1:7]
seed = int(seed_str)
all_ds = all_str.split()
targets = targets_str.split()
force = force_str == "1"

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get_train_path(ds):
    """flowgen_*: flowgen_train_<level>.json;  other: <ds>_arrow_training_all.json"""
    if ds.startswith("flowgen_"):
        level = ds.split("flowgen_", 1)[1]
        return os.path.join(arrow_dir, f"flowgen_train_{level}.json")
    return os.path.join(arrow_dir, f"{ds}_arrow_training_all.json")

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
    """scan output/{model}_arrow_{ds}_bs*/, pick min(eval_loss) corresponding  *global* BS. """
    pattern = os.path.join(output_base, f"{model_key}_arrow_{ds}_bs*")
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
    """from training_logs/{model}_arrow_{ds}_bs{bs}_*.log in grep 'Num Devices: N'. 
    multiple  log take latest timestamp that . """
    pat = os.path.join(log_dir, f"{model_key}_arrow_{ds}_bs{bs}_*.log")
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
log "${CYAN}[Stage 2] training leave-one-out (Arrow) (epoch=${NUM_EPOCHS}, no val) ${NC}"
log ""

TOTAL=0
DONE=0
SKIPPED=0
FAILED=0
while IFS='|' read -r mk tgt bs _; do
    [ -z "$mk" ] && continue
    [ "$bs" = "0" ] && continue
    is_skip_train_target "$tgt" && continue
    TOTAL=$((TOTAL + 1))
done <<< "$BS_MODES"

log "total training tasks: ${TOTAL}    (skip target: ${SKIP_TRAIN_TARGETS[*]})"
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

    RUN_NAME="${model_key}_arrow_loo_${target}_bs${bs_mode}_e${NUM_EPOCHS}"
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
    else
        log "    ${RED}✗ failed (exit=${EC})${NC}"
        tail -8 "$TASK_LOG" 2>/dev/null | while read line; do log "      $line"; done
        FAILED=$((FAILED + 1))
    fi
done <<< "$BS_MODES"

log ""
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log "${BLUE}  done: success=${DONE}  skip=${SKIPPED}  failed=${FAILED}  /  total=${TOTAL}${NC}"

# ============================================================
# ⑨ Stage 3: auto-eval (SKIP_EVAL=1 can skip; ARROW_SOURCES control which arrow source)
# ============================================================
if [ "${SKIP_EVAL:-0}" = "1" ]; then
    log "${YELLOW}SKIP_EVAL=1, skipauto-eval${NC}"
    exit 0
fi

ARROW_SOURCES="${ARROW_SOURCES:-sam3_ft}"
for _src in $ARROW_SOURCES; do
    case "$_src" in
        sam|sam3_ft|groundtruth|yolo) ;;
        *) log "${RED}invalid arrow source '$_src'${NC}"; exit 1 ;;
    esac
done

# ── SAM3 detector (by dataset auto-select best ckpt) ──
DETECTION_RUNS_DIR="<REPO_ROOT>/detection/runs"
find_best_sam3_ckpt() {
    local ds="$1"
    python3 << PYEOF
import re, os, glob
ds = "$ds"
run_dir = os.path.join("$DETECTION_RUNS_DIR", f"arrowhead_{ds}")
if not os.path.isdir(run_dir):
    print(""); exit(0)
logs = sorted(glob.glob(os.path.join(run_dir, "train_*.log")))
best_log, best_count = None, 0
for log in logs:
    c = open(log).read().count("Average Precision")
    if c > best_count:
        best_log, best_count = log, c
if not best_log:
    print(""); exit(0)
content = open(best_log).read()
ap_all = re.findall(r'Average Precision  \(AP\) @\[ IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = ([\d.]+)', content)
ap50  = re.findall(r'Average Precision  \(AP\) @\[ IoU=0\.50      \| area=   all \| maxDets=100 \] = ([\d.]+)', content)
ap75  = re.findall(r'Average Precision  \(AP\) @\[ IoU=0\.75      \| area=   all \| maxDets=100 \] = ([\d.]+)', content)
if not ap_all:
    print(""); exit(0)
val_epochs = list(range(5, 5 * len(ap_all) + 1, 5))
best_avg, best_epoch = -1, -1
for i in range(len(ap_all)):
    avg = (float(ap_all[i]) + float(ap50[i]) + float(ap75[i])) / 3
    if avg > best_avg:
        best_avg, best_epoch = avg, val_epochs[i]
ckpt = os.path.join(run_dir, "checkpoints", f"checkpoint_{best_epoch}.pt")
print(ckpt if os.path.exists(ckpt) else "")
PYEOF
}

declare -A BEST_SAM3_CKPT
_SAM3_DATASETS=("bpmn" "flowgen_easy" "flowgen_medium" "flowgen_hard" "flowlearn" "flowvqa" "fcb")
for _ds in "${_SAM3_DATASETS[@]}"; do
    BEST_SAM3_CKPT["$_ds"]=$(find_best_sam3_ckpt "$_ds")
done
_NO_FT_DATASETS=("fca" "cbd")

# ── eval config ──
EVAL_OUTPUT_DIR="<OUTPUT_ROOT>/output_lora_arrow_loo"
INFERENCE_GPUS="${INFERENCE_GPUS:-$CUDA_DEVICES}"

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
    ["flowgen_easy_orig"]="<DATA_ROOT>/flowgen/test_img_easy_orig"
    ["flowgen_medium_orig"]="<DATA_ROOT>/flowgen/test_img_medium_orig"
    ["flowgen_hard_orig"]="<DATA_ROOT>/flowgen/test_img_hard_orig"
)
declare -A TEST_JSON
TEST_JSON=(
    ["flowlearn"]="<DATA_ROOT>/flowlearn/test.json"
    ["flowvqa"]="<DATA_ROOT>/flowvqa/Data/test_full.json"
)

EVAL_PY_BIN="<REPO_ROOT>/lora/eval/eval_TRACE.py"
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
log "${BLUE}  Targets: ${ALL_DATASETS[*]}${NC}"
log "${BLUE}  Arrow Sources: ${ARROW_SOURCES}${NC}"
log "${BLUE}════════════════════════════════════════════════════════════${NC}"
log ""

# scan LOO training dir
LOO_RUNS=$(python3 - "$OUTPUT_BASE_DIR" "$(echo "${!MODELS[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')" "${ALL_DATASETS[*]}" << 'PYEOF'
import glob, os, re, sys
output_base, models_str, targets_str = sys.argv[1:4]
for mk in models_str.split():
    for tgt in targets_str.split():
        pat = os.path.join(output_base, f"{mk}_arrow_loo_{tgt}_bs*_e*")
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
for _as in $ARROW_SOURCES; do
    while IFS='|' read -r _ _ _ _ _; do
        [ -n "$_" ] && TASK_COUNT=$((TASK_COUNT + 1))
    done <<< "$LOO_RUNS"
done

for ARROW_SOURCE_CUR in $ARROW_SOURCES; do
    log "${BLUE}──── Arrow Source: ${ARROW_SOURCE_CUR} ────${NC}"

    while IFS='|' read -r model_key target bs ckpt_dir step; do
        [ -z "$model_key" ] && continue

        MODEL_PATH="${MODELS[$model_key]}"
        EVAL_DS="${EVAL_DATASET[$target]}"
        T_JSON="${TEST_JSON[$target]:-}"

        if [[ "$target" == flowgen_* && "$ARROW_SOURCE_CUR" != "groundtruth" ]]; then
            IMAGE_DIR="${TEST_IMAGE_DIR[${target}_orig]:-${TEST_IMAGE_DIR[$target]}}"
        else
            IMAGE_DIR="${TEST_IMAGE_DIR[$target]}"
        fi

        if [ -z "$MODEL_PATH" ] || [ -z "$IMAGE_DIR" ] || [ -z "$EVAL_DS" ]; then
            log "${YELLOW}  ⊘ ${model_key}/loo_${target}/${ARROW_SOURCE_CUR}: incomplete config${NC}"
            continue
        fi

        # arrow source fall back
        DS_ARROW_SOURCE="$ARROW_SOURCE_CUR"
        DS_SAM3_CKPT="${BEST_SAM3_CKPT[$target]}"
        for _nft in "${_NO_FT_DATASETS[@]}"; do
            if [[ "$target" == "$_nft" && "$DS_ARROW_SOURCE" == "sam3_ft" ]]; then
                DS_ARROW_SOURCE="sam"
                break
            fi
        done
        if [[ "$DS_ARROW_SOURCE" == "sam3_ft" && -z "$DS_SAM3_CKPT" ]]; then
            log "    ${YELLOW}⚠ ${target} no SAM3 ft detector, fall back sam${NC}"
            DS_ARROW_SOURCE="sam"
        fi

        RESULT_DIR="${EVAL_OUTPUT_DIR}/${model_key}_arrow_loo_${target}_bs${bs}_step${step}_${DS_ARROW_SOURCE}"
        mkdir -p "$RESULT_DIR"

        log "${GREEN}  ▶ [${DONE_COUNT}/${TASK_COUNT}] ${model_key}/loo=${target}/${DS_ARROW_SOURCE}${NC}"
        log "    BS=${bs}  ckpt=${ckpt_dir}"

        # inference
        INFER_CMD="python -m lora.inference.get_triplets_sft_trace"
        INFER_CMD="${INFER_CMD} --output-dir \"${RESULT_DIR}\""
        INFER_CMD="${INFER_CMD} --image-dir \"${IMAGE_DIR}\""
        INFER_CMD="${INFER_CMD} --backend local"
        INFER_CMD="${INFER_CMD} --adapter-path ${ckpt_dir}"
        INFER_CMD="${INFER_CMD} --base-model-path ${MODEL_PATH}"
        INFER_CMD="${INFER_CMD} --gpus ${INFERENCE_GPUS}"
        INFER_CMD="${INFER_CMD} --workers-per-gpu 2"
        INFER_CMD="${INFER_CMD} --arrow-source ${DS_ARROW_SOURCE}"
        [[ "$DS_ARROW_SOURCE" == "sam3_ft" ]] && INFER_CMD="${INFER_CMD} --sam3-ft-checkpoint ${DS_SAM3_CKPT}"
        [ -n "$T_JSON" ] && INFER_CMD="${INFER_CMD} --test-json ${T_JSON}"
        [[ "$target" == flowgen_* ]] && INFER_CMD="${INFER_CMD} --is-flowgen"
        [[ "$target" == "bpmn" ]] && INFER_CMD="${INFER_CMD} --is-bpmn"

        INFER_LOG="${LOG_DIR}/loo_TRACE_infer_${model_key}_${target}_bs${bs}_${DS_ARROW_SOURCE}_${TIMESTAMP}.log"

        EXISTING=$(find "$RESULT_DIR" -name "arrow_triplets.json" -type f 2>/dev/null | wc -l)
        if [ -n "$T_JSON" ] && [ -f "$T_JSON" ]; then
            TOTAL=$(python3 -c "import json; d=json.load(open('$T_JSON')); print(len(d) if isinstance(d,(list,dict)) else 0)" 2>/dev/null)
            [ -z "$TOTAL" ] && TOTAL=0
        else
            TOTAL=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
        fi
        if [ "$EXISTING" -ge "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
            log "    inference already done (${EXISTING}/${TOTAL})"
        else
            log "    inferring... (already have ${EXISTING}/${TOTAL})"
            (cd "<REPO_ROOT>" && eval ${INFER_CMD}) > "${INFER_LOG}" 2>&1
            if [ $? -ne 0 ]; then
                log "    ${RED}✗ inference failed, see ${INFER_LOG}${NC}"
                tail -10 "$INFER_LOG" 2>/dev/null | while read l; do log "      $l"; done
                FAIL_COUNT=$((FAIL_COUNT + 1))
                continue
            fi
        fi

        # scoring
        EVAL_EXTRA=""
        [ -n "${EVAL_GT_ROOT[$target]}" ] && EVAL_EXTRA="--gt-root ${EVAL_GT_ROOT[$target]}"
        EVAL_OUTPUT=$(cd "<REPO_ROOT>" && python "$EVAL_PY_BIN" --dataset "$EVAL_DS" --output-dir "$RESULT_DIR" $EVAL_EXTRA --image-dir "$IMAGE_DIR" --test-json "${T_JSON:-}" 2>&1)
        if [ $? -ne 0 ]; then
            log "    ${RED}✗ scoring failed${NC}"
            echo "$EVAL_OUTPUT" | head -10 | while read l; do log "      $l"; done
            FAIL_COUNT=$((FAIL_COUNT + 1))
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
        FINAL_RESULTS["${model_key}|${target}|${DS_ARROW_SOURCE}"]="${bs}|${step}|${SCORE}"
        DONE_COUNT=$((DONE_COUNT + 1))
    done <<< "$LOO_RUNS"
done

# summary
log ""
log "${BLUE}════════ LOO TRACE eval results ════════${NC}"
log "$(printf '  %-16s %-16s %-12s %-6s %-8s %-10s' 'Model' 'LeaveOut' 'ArrowSrc' 'BS' 'Step' 'F1')"
for key in $(echo "${!FINAL_RESULTS[@]}" | tr ' ' '\n' | sort); do
    IFS='|' read -r mk tgt asrc <<< "$key"
    IFS='|' read -r bs step score <<< "${FINAL_RESULTS[$key]}"
    log "$(printf '  %-16s %-16s %-12s %-6s %-8s %-10s' "$mk" "$tgt" "$asrc" "$bs" "$step" "$score")"
done

SUMMARY_JSON="${EVAL_OUTPUT_DIR}/loo_TRACE_eval_${TIMESTAMP}.json"
python3 -c "
import json
res = {}
raw = '''$(for k in "${!FINAL_RESULTS[@]}"; do echo "${k}=${FINAL_RESULTS[$k]}"; done)'''
for line in raw.strip().split('\n'):
    if not line.strip(): continue
    k, v = line.split('=', 1)
    mk, tgt, asrc = k.split('|')
    bs, step, score = v.split('|')
    res[f'{mk}/loo_{tgt}/{asrc}'] = {'batch_size': int(bs), 'step': int(step), 'test_f1': score}
json.dump(res, open('$SUMMARY_JSON', 'w'), indent=2, ensure_ascii=False)
print('results saved:', '$SUMMARY_JSON')
" 2>&1 | while read l; do log "  $l"; done

log ""
log "  eval: done=${DONE_COUNT}  failed=${FAIL_COUNT}  total=${TASK_COUNT}"
[ "$FAIL_COUNT" -gt 0 ] && exit 1 || exit 0
