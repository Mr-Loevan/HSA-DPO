#!/bin/bash

# HSA-DPO Training Script for LLaVA-v1.5

# Training configuration
BATCH_SIZE=8
EPOCH=2
LEARNING_RATE=2e-6

# Paths - Update these according to your environment
DATA_PATH="./hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl"
IMAGE_FOLDER="./hsa_dpo/data/image"
MODEL_PATH="./models/llava-v1.5-13b"  # Path to your LLaVA model
VISION_TOWER="openai/clip-vit-large-patch14-336"  # Will be auto-downloaded
OUTPUT_DIR="./output/hsa_dpo_llava"
DS_CONFIG="./hsa_dpo/models/llava-v1_5/scripts/zero3.json"

# Number of GPUs to use
NUM_GPUS=2

# Training script entry point
ENTRY="hsa_dpo/models/llava-v1_5/train_dpo.py"

echo "Starting HSA-DPO training..."
echo "Data path: ${DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Using ${NUM_GPUS} GPUs"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run training with DeepSpeed
deepspeed --num_gpus=${NUM_GPUS} ${ENTRY} \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --desc_data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 0 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --deepspeed ${DS_CONFIG} \
    --beta 0.1 \
    --use_chosen_score False \
    --use_rejected_score True

echo "Training completed!"