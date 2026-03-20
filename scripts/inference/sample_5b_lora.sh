#!/usr/bin/bash

export TOKENIZERS_PARALLELISM=false

LORA_CKPT="ckpts/YUME/outputs_yume1.5_stage1/lora-checkpoint-200"
LORA_RANK=128
LORA_ALPHA=256

### I2V + LoRA
torchrun --nproc_per_node 8 --master_port 9601 \
    fastvideo/sample/sample_5b.py \
    --seed 43 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_sample_steps=600000 \
    --mixed_precision="bf16" \
    --allow_tf32 \
    --video_output_dir="./outputs/test_lora" \
    --caption_path="./caption_re.txt" \
    --test_data_dir="./val" \
    --num_euler_timesteps 50 \
    --rand_num_img 0.6 \
    --jpg_dir="./jpg/" \
    --prompt "A fire-breathing dragon appeared." \
    --use_lora \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --resume_from_checkpoint "${LORA_CKPT}"

# num_euler_timesteps  stage1 50 steps. stage2.3 4 steps.
### T2V + LoRA
torchrun --nproc_per_node 8 --master_port 9601 \
    fastvideo/sample/sample_5b.py \
    --seed 43 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_sample_steps=600000 \
    --mixed_precision="bf16" \
    --allow_tf32 \
    --video_output_dir="./outputs/test_lora" \
    --caption_path="./caption_re.txt" \
    --test_data_dir="./val" \
    --num_euler_timesteps 50 \
    --rand_num_img 0.6 \
    --T2V \
    --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage." \
    --use_lora \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --resume_from_checkpoint "${LORA_CKPT}"

### ── Stage 2/3 (full checkpoint, MVDT, 4 steps) ──────────────────────────
# Stage 2/3 saves full checkpoints (not LoRA), so no --use_lora needed.
# STAGE23_CKPT="outputs_train_yume1.5_stage23/TIMESTAMP/checkpoint-STEP"

## I2V
# torchrun --nproc_per_node 8 --master_port 9601 \
#     fastvideo/sample/sample_5b.py \
#     --seed 43 \
#     --gradient_checkpointing \
#     --train_batch_size=1 \
#     --max_sample_steps=600000 \
#     --mixed_precision="bf16" \
#     --allow_tf32 \
#     --video_output_dir="./outputs/stage23_test" \
#     --caption_path="./caption_re.txt" \
#     --test_data_dir="./val" \
#     --num_euler_timesteps 4 \
#     --rand_num_img 0.6 \
#     --MVDT \
#     --jpg_dir="./jpg/" \
#     --prompt "A fire-breathing dragon appeared." \
#     --resume_from_checkpoint "${STAGE23_CKPT}"

## T2V
# torchrun --nproc_per_node 8 --master_port 9601 \
#     fastvideo/sample/sample_5b.py \
#     --seed 43 \
#     --gradient_checkpointing \
#     --train_batch_size=1 \
#     --max_sample_steps=600000 \
#     --mixed_precision="bf16" \
#     --allow_tf32 \
#     --video_output_dir="./outputs/stage23_test" \
#     --caption_path="./caption_re.txt" \
#     --test_data_dir="./val" \
#     --num_euler_timesteps 4 \
#     --rand_num_img 0.6 \
#     --MVDT \
#     --T2V \
#     --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage." \
#     --resume_from_checkpoint "${STAGE23_CKPT}"
