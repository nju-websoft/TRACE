#!/bin/bash

# ============================================================
# train_early_stop.sh (LLaVa-V1.6-Mistral-7B-hf, ms-swift)
#
# single run training entry (with val + early stopping) . CLI vs Qwen-VL-Series-Finetune
# train_early_stop.sh align.
# ============================================================

# ==================== default params ====================
MODEL_NAME="<MODEL_ROOT>/llava-v1.6-mistral-7b-hf"
CUDA_DEVICES="0,1,2,3,4,5,6,7"
DATA_PATH=""
EVAL_DATA_PATH=""
OUTPUT_DIR=""
NUM_EPOCHS=10
BATCH_SIZE=8
NUM_DEVICES=8
SAVE_INTERVAL=1.0

EARLY_STOPPING_PATIENCE=3
EARLY_STOPPING_THRESHOLD=0.0
EVAL_INTERVAL=1.0
METRIC_FOR_BEST_MODEL="eval_loss"

# swift / modelrelated
MODEL_TYPE="llava1_6_mistral_hf"
MAX_LENGTH=4096
LEARNING_RATE=2e-4
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
DEEPSPEED_STAGE="zero3"   # canchange zero3, flowvqa / flowgen_hard such long sequences + LLaVa vision token saves memory when large

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

generic interface ( vs  Qwen-VL-Series-Finetune/scripts/train_early_stop.sh align) :
  --model_name PATH               base model path
  --cuda_devices "0,1,..."        visible GPU
  --data_path PATH                training JSON
  --eval_data_path PATH           validation JSON (early stoppingrequired) 
  --output_dir DIR                output dir
  --num_epochs N                  at mosttraining epoch
  --batch_size N                  global batch size
  --num_devices N                 GPU count
  --save_interval F               (compat field)
  --early_stopping_patience N
  --early_stopping_threshold F
  --metric_for_best_model STR     default ${METRIC_FOR_BEST_MODEL}

Swift specific:
  --model_type STR / --max_length N / --learning_rate F
  --lora_rank N / --lora_alpha N / --lora_dropout F
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --cuda_devices) CUDA_DEVICES="$2"; shift 2 ;;
        --data_path) DATA_PATH="$2"; shift 2 ;;
        --eval_data_path) EVAL_DATA_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --num_devices) NUM_DEVICES="$2"; shift 2 ;;
        --save_interval) SAVE_INTERVAL="$2"; shift 2 ;;
        --early_stopping_patience) EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
        --early_stopping_threshold) EARLY_STOPPING_THRESHOLD="$2"; shift 2 ;;
        --eval_interval) EVAL_INTERVAL="$2"; shift 2 ;;
        --metric_for_best_model) METRIC_FOR_BEST_MODEL="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --max_length) MAX_LENGTH="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --lora_rank) LORA_RANK="$2"; shift 2 ;;
        --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --deepspeed_stage) DEEPSPEED_STAGE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[ -z "$DATA_PATH" ]      && { echo "Error: --data_path is required"; usage; }
[ -z "$EVAL_DATA_PATH" ] && { echo "Error: --eval_data_path is required for early stopping"; usage; }
[ -z "$OUTPUT_DIR" ]     && { echo "Error: --output_dir is required"; usage; }

BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=$((BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
[ $GRAD_ACCUM_STEPS -lt 1 ] && GRAD_ACCUM_STEPS=1

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export NPROC_PER_NODE=$NUM_DEVICES

# ============================================================
# NCCL / signal cleanup: training on abnormal exit, try to let communicator release
# ============================================================
export NCCL_ASYNC_ERROR_HANDLING=1   # let NCCL exit immediately on error abort instead of hanging
export NCCL_TIMEOUT=600              # 10 min timeout (default 30 min too long, hangs long) 
# same process group trap: when parent script is interrupted, take down all worker as well
_cleanup() {
    echo "[trap] on signal, kill this process group  swift / torchrun ..."
    pkill -P $$ 2>/dev/null || true
    pkill -u "$USER" -f "swift/cli/sft.py" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

echo "========================================="
echo "LLaVa-V1.6-Mistral-7B-hf Training Configuration (Early Stopping):"
echo "========================================="
echo "Model Name:           $MODEL_NAME"
echo "Model Type:           $MODEL_TYPE"
echo "CUDA Devices:         $CUDA_DEVICES"
echo "Data Path:            $DATA_PATH"
echo "Eval Data Path:       $EVAL_DATA_PATH"
echo "Output Dir:           $OUTPUT_DIR"
echo "Max Epochs:           $NUM_EPOCHS"
echo "Global Batch Size:    $BATCH_SIZE"
echo "Batch Per Device:     $BATCH_PER_DEVICE"
echo "Num Devices:          $NUM_DEVICES"
echo "Grad Accum Steps:     $GRAD_ACCUM_STEPS"
echo "-----------------------------------------"
echo "Early Stopping:"
echo "  Patience:           $EARLY_STOPPING_PATIENCE"
echo "  Threshold:          $EARLY_STOPPING_THRESHOLD"
echo "  Save/Eval Strategy: epoch"
echo "  Monitor Metric:     $METRIC_FOR_BEST_MODEL"
echo "========================================="
echo ""


# ============================================================
# pre-check 1: GPU health check + clean up leftover processes
# crash source is not single : previous round swift training crash / by Ctrl+C when, NCCL
# communicator won't be reap dropped, will at rank 0 GPU leave on 5×578MB
# ghost process; next round resume then one more stacked run can OOM that card (driver-level
# "Not Supported") . here proactively kill all of this user's swift /
# torchrun leftover + check CUDA_VISIBLE_DEVICES entries' whether the card is alive.
# ============================================================
echo "[pre-flight] kill this user's leftovers  swift / torchrun process..."
pkill -u "$USER" -9 -f "swift/cli/sft.py"          2>/dev/null || true
pkill -u "$USER" -9 -f "torch/distributed/run.*swift" 2>/dev/null || true
pkill -u "$USER" -9 -f "torchrun.*swift"           2>/dev/null || true
sleep 3   # give time for NCCL communicator release

# health check: CVD in whether each card can be nvidia-smi in 5 get status within seconds
echo "[pre-flight] check CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}  health..."
IFS=',' read -ra _CHECK_GPUS <<< "$CUDA_DEVICES"
for _g in "${_CHECK_GPUS[@]}"; do
    if ! timeout 5 nvidia-smi -i "$_g" --query-gpu=name,utilization.gpu --format=csv,noheader >/dev/null 2>&1; then
        echo "[pre-flight] ✗ GPU $_g unresponsive or driver-level hang. "
        echo "             try: sudo nvidia-smi --gpu-reset -i $_g  (need sudo)"
        echo "              or : wait a few minutes for D-state processes to release / reboot the machine"
        echo "             can also temporarily use CUDA_DEVICES=\"<canuse GPU>\" NUM_DEVICES=N runthis round"
        exit 1
    fi
done
echo "[pre-flight] alltarget GPU responsive"

# ============================================================
# pre-check 2: resume checkpoint
# swift to_abspath validates that paths exist; 'last' sentinel not recognized.
# ============================================================
RESUME_ARG=()
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CKPT=$( { ls -1d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null; \
                     ls -1d "$OUTPUT_DIR"/v*-*/checkpoint-* 2>/dev/null; } \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -n \
        | tail -n1 \
        | awk '{print $2}')
    if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
        # complete resume (resume optimizer / lr scheduler / early stopping patience)
        # Note: multi-GPU resume when swift makes all rank in rank0 deserialize on card optimizer
        # state, may stack up ghost processes OOM that card. ifthen hit this issue, the line below
        # change back (--resume_from_checkpoint "$LATEST_CKPT" --resume_only_model true)
        RESUME_ARG=(--resume_from_checkpoint "$LATEST_CKPT")
        echo "Resuming from: $LATEST_CKPT  (full resume - optimizer/lr/patience all restored)"
    fi
fi

# random MASTER_PORT, avoid grid_search previous run's port not released during serial runs
if [ -z "${MASTER_PORT:-}" ]; then
    MASTER_PORT=$(( 20000 + RANDOM % 10000 ))
fi
export MASTER_PORT
echo "[DDP] MASTER_PORT=${MASTER_PORT}"

# ── LLaVA → SWIFT messages format conversion ──
# swift sft need messages/role/content + images format, whiletraining datais LLaVA
# conversations/from/value + image format, convert on the fly and cache to output_dir.
CONVERT_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/convert_to_swift_format.py"
_SWIFT_CACHE="${OUTPUT_DIR}/_swift_data"
mkdir -p "$_SWIFT_CACHE"
_DATA_SWIFT="${_SWIFT_CACHE}/train_swift.json"
python3 "$CONVERT_SCRIPT" --input "$DATA_PATH" --output "$_DATA_SWIFT"
DATA_PATH="$_DATA_SWIFT"
if [ -n "$EVAL_DATA_PATH" ]; then
    _EVAL_SWIFT="${_SWIFT_CACHE}/val_swift.json"
    python3 "$CONVERT_SCRIPT" --input "$EVAL_DATA_PATH" --output "$_EVAL_SWIFT"
    EVAL_DATA_PATH="$_EVAL_SWIFT"
fi

swift sft \
    --model "$MODEL_NAME" \
    --model_type "$MODEL_TYPE" \
    --tuner_type lora \
    --dataset "$DATA_PATH" \
    --val_dataset "$EVAL_DATA_PATH" \
    --torch_dtype bfloat16 \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_PER_DEVICE" \
    --per_device_eval_batch_size "$BATCH_PER_DEVICE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --target_modules all-linear \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --max_length "$MAX_LENGTH" \
    --save_strategy epoch \
    --save_total_limit 100 \
    --eval_strategy epoch \
    --metric_for_best_model "$METRIC_FOR_BEST_MODEL" \
    --greater_is_better false \
    --load_best_model_at_end true \
    --early_stop_interval "$EARLY_STOPPING_PATIENCE" \
    --logging_steps 1 \
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --attn_impl flash_attn \
    --gradient_checkpointing false \
    --vit_gradient_checkpointing false \
    --deepspeed "$DEEPSPEED_STAGE" \
    "${RESUME_ARG[@]}"

echo ""
echo "Training completed for $OUTPUT_DIR"
