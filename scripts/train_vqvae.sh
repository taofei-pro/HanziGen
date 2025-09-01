#!/bin/bash

TARGET_FONT_PATH="fonts/M8.ttf"
TRAIN_SPLIT_RATIO=0.8
VAL_SPLIT_RATIO=0.2
RANDOM_SEED=2025
BATCH_SIZE=16           # 从32降到16，平衡显存和速度
LEARNING_RATE=5e-4      # 保持5e-4
NUM_EPOCHS=150          # 从200降到150，平衡训练时间
DEVICE="cuda"


TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

MODEL_SAVE_PATH="checkpoints/vqvae_${TARGET_FONT_NAME}.pth"

python train_vqvae.py \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --device "$DEVICE"
