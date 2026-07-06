#!/bin/bash
set -e

: "${NUM_GPUS:=8}"
: "${DATA_PATH:=./training_data}"
: "${OUTPUT_DIR:=./userlm_checkpoints}"
: "${MODEL_NAME:=Qwen/Qwen2.5-14B-Instruct}"
: "${TOTAL_BS:=64}"
: "${GRAD_ACC:=8}"
: "${USE_TORCH_COMPILE:=True}"
: "${USE_PROFILER:=False}"
: "${PROFILER_DIR:=./profiler_results}"
: "${RUN_VALIDATION_BEFORE_TRAIN:=True}"


echo "========================================"
echo "UserLM Multi-GPU Training (FSDP + LoRA)"
echo "========================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Model name: $MODEL_NAME"
echo "Total batch size: $TOTAL_BS"
echo "Gradient accumulation steps: $GRAD_ACC"
echo "Per-GPU batch size: $((TOTAL_BS / NUM_GPUS / GRAD_ACC))"
echo ""

MODEL_NAME_STRING=${MODEL_NAME//\//--}
echo "Model name string for saving: $MODEL_NAME_STRING"

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data path $DATA_PATH does not exist"
    exit 1
fi

if [ ! -f "$DATA_PATH/train_${MODEL_NAME_STRING}_samples.jsonl" ] || \
   [ ! -f "$DATA_PATH/val_${MODEL_NAME_STRING}_samples.jsonl" ]   || \
   [ ! -f "$DATA_PATH/test_${MODEL_NAME_STRING}_samples.jsonl" ]; then
    echo "Error: One or more training data files not found in $DATA_PATH"
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ $AVAILABLE_GPUS -lt $NUM_GPUS ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi

# Run distributed training
torchrun --nnodes 1 \
    --nproc_per_node $NUM_GPUS \
    --master_port $((29500 + RANDOM % 1000)) \
    5_train_userlm.py \
    --tokenizer_name "./tokenizers_and_configs--$MODEL_NAME_STRING" \
    --enable_fsdp \
    --fsdp_config.fsdp_activation_checkpointing True \
    --fsdp_config.pure_bf16 \
    --use_peft=False \
    --use_fast_kernels \
    --checkpoint_type StateDictType.SHARDED_STATE_DICT \
    --peft_method='lora' \
    --mixed_precision \
    --batch_size_training $((TOTAL_BS / NUM_GPUS / GRAD_ACC)) \
    --val_batch_size $((TOTAL_BS / NUM_GPUS / GRAD_ACC)) \
    --gradient_accumulation_steps $GRAD_ACC \
    --dist_checkpoint_root_folder "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batching_strategy='padding' \
    --dataset userlm_dataset \
    --dataset_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --lr 2e-5 \
    --num_epochs 2 \
    --weight_decay 0.0 \
    --gamma 0.85 \
    --lora_config.r 8 \
    --lora_config.lora_alpha 32 \
    --use_wandb True \
    --wandb_config.project YOUR_PROJECT \
    --wandb_config.entity YOUR_ENTITY \
    --wandb_config.name sftuser_training-$MODEL_NAME_STRING \
    --run_validation True \
    --eval_steps 100 \
    --train_config.run_validation_before_train $RUN_VALIDATION_BEFORE_TRAIN \
    --warmup_ratio 0.1 \
    --min_lr 2e-6 \
    --use_torch_compile $USE_TORCH_COMPILE \
    --use_profiler $USE_PROFILER \
    --profiler_dir "$PROFILER_DIR"
