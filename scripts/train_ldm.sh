#!/bin/bash

TARGET_FONT_PATH="fonts/M8.ttf"
TRAIN_SPLIT_RATIO=0.9           # 保持0.9/0.1比例
VAL_SPLIT_RATIO=0.1             # 保持0.9/0.1比例
RANDOM_SEED=2025
BATCH_SIZE=32                    # 从28增加到32，小幅提升训练效率
LEARNING_RATE=2e-4               # 从2.5e-4调到2e-4，更精细的学习率
NUM_EPOCHS=500                   # 从450增加到500，配合早停机制
SAMPLE_STEPS=120                 # 从100增加到120，提升生成质量
IMG_SAVE_INTERVAL=5              # 保持5
LPIPS_EVAL_INTERVAL=10           # 保持10
EVAL_BATCH_SIZE=4                # 保持4
DEVICE="cuda"


TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

PRETRAINED_VQVAE_PATH="checkpoints/vqvae_${TARGET_FONT_NAME}.pth"
MODEL_SAVE_PATH="checkpoints/ldm_${TARGET_FONT_NAME}.pth"
SAMPLE_ROOT="samples_${TARGET_FONT_NAME}/"

python train_ldm.py \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --pretrained_vqvae_path "$PRETRAINED_VQVAE_PATH" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --sample_root "$SAMPLE_ROOT" \
    --sample_steps "$SAMPLE_STEPS" \
    --img_save_interval "$IMG_SAVE_INTERVAL" \
    --lpips_eval_interval "$LPIPS_EVAL_INTERVAL" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE"
