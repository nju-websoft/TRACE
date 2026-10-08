#!/bin/bash

# train_early_stop.sh
# MiniCPM-V4.5-8B LoRA SFT with early stopping via SWIFT framework.
#
# SWIFT uses HuggingFace Trainer under the hood, so eval_strategy /
# load_best_model_at_end / early_stopping_patience all work.

# ==================== default params ====================
MODEL_NAME="/path/to/MiniCPM-V-4_5"   # TODO: replace after download
CUDA_DEVICES="0,1,2,3,4,5,6,7"
DATA_PATH=""
EVAL_DATA_PATH=""
OUTPUT_DIR=""
NUM_EPOCHS=10
BATCH_SIZE=8
NUM_DEVICES=8

# LoRA args
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.05

# early stopping params
EARLY_STOPPING_PATIENCE=3
EARLY_STOPPING_THRESHOLD=0.0
METRIC_FOR_BEST_MODEL="eval_loss"

# ==================== usage ====================
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model_name MODEL              Model path (default: placeholder)"
    echo "  --cuda_devices DEVICES          CUDA visible devices (default: 0,1)"
    echo "  --data_path PATH                Training data JSON (SWIFT format, required)"
    echo "  --eval_data_path PATH           Eval data JSON (SWIFT format, required)"
    echo "  --output_dir DIR                Output directory (required)"
    echo "  --num_epochs N                  Max training epochs (default: 10)"
    echo "  --batch_size N                  Global batch size (default: 8)"
    echo "  --num_devices N                 Number of GPUs (default: 2)"
    echo ""
    echo "Early stopping options:"
    echo "  --early_stopping_patience N     (default: 3)"
    echo "  --early_stopping_threshold F    (default: 0.0)"
    exit 1
}

# ==================== parse args ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)           MODEL_NAME="$2";                shift 2 ;;
        --cuda_devices)         CUDA_DEVICES="$2";              shift 2 ;;
        --data_path)            DATA_PATH="$2";                 shift 2 ;;
        --eval_data_path)       EVAL_DATA_PATH="$2";            shift 2 ;;
        --output_dir)           OUTPUT_DIR="$2";                shift 2 ;;
        --num_epochs)           NUM_EPOCHS="$2";                shift 2 ;;
        --batch_size)           BATCH_SIZE="$2";                shift 2 ;;
        --num_devices)          NUM_DEVICES="$2";               shift 2 ;;
        --early_stopping_patience)   EARLY_STOPPING_PATIENCE="$2";  shift 2 ;;
        --early_stopping_threshold)  EARLY_STOPPING_THRESHOLD="$2"; shift 2 ;;
        --lora_rank)            LORA_RANK="$2";                 shift 2 ;;
        --lora_alpha)           LORA_ALPHA="$2";                shift 2 ;;
        -h|--help)              usage ;;
        *)                      echo "Unknown option: $1"; usage ;;
    esac
done

# ==================== check required args ====================
if [ -z "$DATA_PATH" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$EVAL_DATA_PATH" ]; then
    echo "Error: --data_path, --eval_data_path, --output_dir are required"
    usage
fi

# ==================== environment setup ====================
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# ==================== compute training params ====================
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=$((BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
if [ "$GRAD_ACCUM_STEPS" -lt 1 ]; then
    GRAD_ACCUM_STEPS=1
fi

# ==================== print config ====================
echo "========================================="
echo "MiniCPM-V4.5 Training (SWIFT + Early Stop)"
echo "========================================="
echo "Model:                $MODEL_NAME"
echo "CUDA Devices:         $CUDA_DEVICES"
echo "Data Path:            $DATA_PATH"
echo "Eval Data Path:       $EVAL_DATA_PATH"
echo "Output Dir:           $OUTPUT_DIR"
echo "Max Epochs:           $NUM_EPOCHS"
echo "Global Batch Size:    $BATCH_SIZE"
echo "Batch Per Device:     $BATCH_PER_DEVICE"
echo "Num Devices:          $NUM_DEVICES"
echo "Grad Accum Steps:     $GRAD_ACCUM_STEPS"
echo "LoRA:                 rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "-----------------------------------------"
echo "Early Stopping:"
echo "  Patience:           $EARLY_STOPPING_PATIENCE"
echo "  Threshold:          $EARLY_STOPPING_THRESHOLD"
echo "  Monitor Metric:     $METRIC_FOR_BEST_MODEL"
echo "========================================="
echo ""

# ==================== start training ====================
# probe $OUTPUT_DIR inmostnew checkpoint; if existsinthen resume, elsenot passed resume args
# compatible with both layouts:
# - Qwen / flat: $OUTPUT_DIR/checkpoint-N
# - MiniCPM/SWIFT: $OUTPUT_DIR/v{K}-{timestamp}/checkpoint-N
RESUME_CKPT=""
if [ -d "$OUTPUT_DIR" ]; then
    LATEST=$( { ls -1d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null; \
                ls -1d "$OUTPUT_DIR"/v*-*/checkpoint-* 2>/dev/null; } \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -n \
        | tail -n1 \
        | awk '{print $2}')
    if [ -n "$LATEST" ] && [ -d "$LATEST" ]; then
        RESUME_CKPT="$LATEST"
        echo "[Resume] already found a checkpoint: $RESUME_CKPT"
    else
        echo "[Resume] noalready have checkpoint, onstart freshnewtraining"
    fi
fi

RESUME_ARG=()
if [ -n "$RESUME_CKPT" ]; then
    RESUME_ARG=(--resume_from_checkpoint "$RESUME_CKPT")
fi

# random MASTER_PORT avoid vs other torchrun conflict (default 29500 inserial grid search when
# easily due to previous run not cleaned up, causing "Address already in use")
if [ -z "${MASTER_PORT:-}" ]; then
    MASTER_PORT=$(( 20000 + RANDOM % 10000 ))
fi
export MASTER_PORT
echo "[DDP] MASTER_PORT=${MASTER_PORT}"

# NPROC_PER_NODE controlmultiple GPU training
NPROC_PER_NODE=$NUM_DEVICES \
MASTER_PORT=$MASTER_PORT \
swift sft \
    --model "$MODEL_NAME" \
    --tuner_type lora \
    --dataset "$DATA_PATH" \
    --val_dataset "$EVAL_DATA_PATH" \
    --torch_dtype bfloat16 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --target_modules all-linear \
    --max_length 4096 \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 100 \
    --eval_strategy epoch \
    --load_best_model_at_end true \
    --metric_for_best_model $METRIC_FOR_BEST_MODEL \
    --greater_is_better false \
    --early_stop_interval $EARLY_STOPPING_PATIENCE \
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --attn_impl flash_attn \
    --model_type minicpmv4_5 \
    --gradient_checkpointing false \
    --vit_gradient_checkpointing false \
    --deepspeed zero3 \
    "${RESUME_ARG[@]}"

echo ""
echo "Training completed for $OUTPUT_DIR"
