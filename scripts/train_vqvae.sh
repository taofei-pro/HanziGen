#!/bin/bash

TARGET_FONT_PATH="fonts/M8.ttf"
TRAIN_SPLIT_RATIO=0.9           # 保持0.9/0.1比例
VAL_SPLIT_RATIO=0.1             # 保持0.9/0.1比例
RANDOM_SEED=2025
BATCH_SIZE=16                    # 从12增加到16，提升训练强度
LEARNING_RATE=6e-4               # 从8e-4调到6e-4，更精细的学习率
NUM_EPOCHS=200                   # 从120增加到200，充分训练
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
