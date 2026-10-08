#!/bin/bash

# train_early_stop.sh
# based on train_triplet.sh, add early stopping (Early Stopping) support.
# must provide --eval_data_path validation set, trainingperiodicallyevaluation eval_loss,
# if consecutive patience timesevaluationstop early if no improvementtraining.

# ==================== default params ====================
MODEL_NAME="<MODEL_ROOT>/Qwen3-VL-4B-Instruct"
CUDA_DEVICES="0,1,2,3,4,5,6,7"
DATA_PATH=""
EVAL_DATA_PATH=""
OUTPUT_DIR=""
NUM_EPOCHS=10
BATCH_SIZE=8
NUM_DEVICES=8
SAVE_INTERVAL=1.0

# early stopping params
EARLY_STOPPING_PATIENCE=3         # how many consecutiveevaluationstop after no improvement
EARLY_STOPPING_THRESHOLD=0.0      # min improvement
EVAL_INTERVAL=1.0                 # every how many  epoch evaluation once ( vs  save_interval align) 
METRIC_FOR_BEST_MODEL="eval_loss" # monitor metric

# ==================== usage ====================
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model_name MODEL              Model name or path (default: "<MODEL_ROOT>/Qwen3-VL-4B-Instruct")"
    echo "  --cuda_devices DEVICES          CUDA visible devices (default: 0,1)"
    echo "  --data_path PATH                Training data JSON path (required)"
    echo "  --eval_data_path PATH           Evaluation data JSON path (required for early stopping)"
    echo "  --output_dir DIR                Output directory (required)"
    echo "  --num_epochs N                  Maximum training epochs (default: 10)"
    echo "  --batch_size N                  Global batch size (default: 8)"
    echo "  --num_devices N                 Number of devices (default: 2)"
    echo ""
    echo "Early stopping options:"
    echo "  --early_stopping_patience N     Stop after N evals with no improvement (default: 3)"
    echo "  --early_stopping_threshold F    Min improvement to count (default: 0.0)"
    echo "  --metric_for_best_model METRIC  Metric to monitor (default: eval_loss)"
    echo ""
    echo "Note: save and eval are always per-epoch (strategy=epoch)"
    echo ""
    echo "  -h, --help                      Show this help message"
    exit 1
}

# ==================== parse args ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --cuda_devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --eval_data_path)
            EVAL_DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_devices)
            NUM_DEVICES="$2"
            shift 2
            ;;
        --save_interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --early_stopping_patience)
            EARLY_STOPPING_PATIENCE="$2"
            shift 2
            ;;
        --early_stopping_threshold)
            EARLY_STOPPING_THRESHOLD="$2"
            shift 2
            ;;
        --eval_interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --metric_for_best_model)
            METRIC_FOR_BEST_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# ==================== check required args ====================
if [ -z "$DATA_PATH" ]; then
    echo "Error: --data_path is required"
    usage
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output_dir is required"
    usage
fi

# --eval_data_path is optional : falls back to plain when omittedtraining (no eval/no early stopping)
if [ -z "$EVAL_DATA_PATH" ]; then
    NO_EVAL_MODE=1
else
    NO_EVAL_MODE=0
fi

# ==================== environment setup ====================
python3 -c "import torch; print(torch.cuda.device_count())"
echo $CUDA_DEVICES
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
python3 -c "import torch; print(torch.cuda.device_count())"

export PYTHONPATH=src:$PYTHONPATH

# ==================== compute training params ====================
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=$((BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# image pixel limits
IMAGE_MIN_PIXELS=$((256 * 28 * 28))
IMAGE_MAX_PIXELS=$((1280 * 28 * 28))

# ==================== print config ====================
echo "========================================="
echo "Training Configuration (Early Stopping):"
echo "========================================="
echo "Model Name:           $MODEL_NAME"
echo "CUDA Devices:         $CUDA_DEVICES"
echo "Data Path:            $DATA_PATH"
echo "Eval Data Path:       $EVAL_DATA_PATH"
echo "Output Dir:           $OUTPUT_DIR"
echo "Max Epochs:           $NUM_EPOCHS"
echo "Global Batch Size:    $BATCH_SIZE"
echo "Batch Per Device:     $BATCH_PER_DEVICE"
echo "Num Devices:          $NUM_DEVICES"
echo "Grad Accum Steps:     $GRAD_ACCUM_STEPS"
echo "Image Pixels:         min=${IMAGE_MIN_PIXELS}, max=${IMAGE_MAX_PIXELS}"
echo "-----------------------------------------"
if [ "$NO_EVAL_MODE" -eq 1 ]; then
    echo "Mode: PLAIN training (no eval/no early stopping, --eval_data_path not provided)"
else
    echo "Early Stopping:"
    echo "  Patience:           $EARLY_STOPPING_PATIENCE"
    echo "  Threshold:          $EARLY_STOPPING_THRESHOLD"
    echo "  Save/Eval Strategy: epoch"
    echo "  Monitor Metric:     $METRIC_FOR_BEST_MODEL"
fi
echo "========================================="
echo ""

# ==================== start training ====================
# common args
COMMON_ARGS=(
    --use_liger_kernel False
    --lora_enable True
    --vision_lora True
    --use_dora False
    --lora_namespan_exclude "['lm_head', 'embed_tokens']"
    --lora_rank 32
    --lora_alpha 64
    --lora_dropout 0.05
    --num_lora_modules -1
    --deepspeed scripts/zero3.json
    --model_id "$MODEL_NAME"
    --data_path "$DATA_PATH"
    --image_folder /path/to/your/image/folder
    --remove_unused_columns False
    --freeze_vision_tower True
    --freeze_llm True
    --freeze_merger True
    --bf16 True
    --fp16 False
    --disable_flash_attn2 False
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs "$NUM_EPOCHS"
    --per_device_train_batch_size $BATCH_PER_DEVICE
    --per_device_eval_batch_size $BATCH_PER_DEVICE
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS
    --image_min_pixels $IMAGE_MIN_PIXELS
    --image_max_pixels $IMAGE_MAX_PIXELS
    --learning_rate 2e-4
    --weight_decay 0.1
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --tf32 True
    --gradient_checkpointing False
    --report_to tensorboard
    --lazy_preprocess True
    --save_strategy "epoch"
    --save_total_limit 100
    --dataloader_num_workers 4
    --resume_from_checkpoint last
)

# mode-specific args
if [ "$NO_EVAL_MODE" -eq 1 ]; then
    EVAL_ARGS=(
        --eval_strategy "no"
        --early_stopping_patience 0
    )
else
    EVAL_ARGS=(
        --eval_path "$EVAL_DATA_PATH"
        --eval_strategy "epoch"
        --load_best_model_at_end True
        --metric_for_best_model "$METRIC_FOR_BEST_MODEL"
        --greater_is_better False
        --early_stopping_patience $EARLY_STOPPING_PATIENCE
        --early_stopping_threshold $EARLY_STOPPING_THRESHOLD
    )
fi

deepspeed src/train/train_sft_early_stop.py "${COMMON_ARGS[@]}" "${EVAL_ARGS[@]}"

echo ""
echo "Training completed for $OUTPUT_DIR"
