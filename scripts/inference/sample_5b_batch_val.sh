#!/usr/bin/bash

export TOKENIZERS_PARALLELISM=false

TSV="/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/data/seadance2_yume_test_12classv2/world_model_action12_test_240_random_order.tsv"
FIRST_FRAME_DIR="/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/data/seadance2_yume_test_12classv2/first_frame"

### ── Stage1 (full checkpoint) ──────────────────────────────────────────────
# CKPT="ckpts/YUME/outputs_yume1.5_stage1_3.4/20260305_003617/checkpoint-400"
# OUTPUT_DIR="./outputs/stage1_val_batch400step_autocaption_3.11"

# torchrun --nproc_per_node 8 --master_port 9602 \
#     fastvideo/sample/sample_5b_batch_val.py \
#     --seed 43 \
#     --gradient_checkpointing \
#     --train_batch_size=1 \
#     --max_sample_steps=600000 \
#     --mixed_precision="bf16" \
#     --allow_tf32 \
#     --video_output_dir="${OUTPUT_DIR}" \
#     --tsv_path="${TSV}" \
#     --first_frame_dir="${FIRST_FRAME_DIR}" \
#     --num_euler_timesteps 50 \
#     --rand_num_img 0.6 \
#     --skip_vlm \
#     --auto_caption \
#     --auto_caption_steps 12 \
#     --num_samples_per_class 5 \
#     --resume_from_checkpoint "${CKPT}"  #去掉就是加载原版

### ── Stage1 + LoRA ──────────────────────────────────────────────────────────
LORA_CKPT="/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/ckpts/YUME/outputs_yume1.5_stage1_lora_seadancev2/20260314_235554/lora-checkpoint-1000"
LORA_RANK=128
LORA_ALPHA=128
OUTPUT_DIR_LORA="./outputs/stage1_lora_valv2_batch_3.15_1000steps"

torchrun --nproc_per_node 8 --master_port 9603 \
    fastvideo/sample/sample_5b_batch_val.py \
    --seed 43 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_sample_steps=600000 \
    --mixed_precision="bf16" \
    --allow_tf32 \
    --video_output_dir="${OUTPUT_DIR_LORA}" \
    --tsv_path="${TSV}" \
    --first_frame_dir="${FIRST_FRAME_DIR}" \
    --num_euler_timesteps 50 \
    --rand_num_img 0.6 \
    --skip_vlm \
    --auto_caption \
    --auto_caption_steps 12 \
    --num_samples_per_class 5 \
    --use_lora \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --resume_from_checkpoint "${LORA_CKPT}"


# --num_samples_per_class 5 \ # 每个类生成5个样本