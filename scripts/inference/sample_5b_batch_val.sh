#!/usr/bin/bash

export TOKENIZERS_PARALLELISM=false

TSV="/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/data/seadance2_yume_test_12classv2/world_model_action12_test_240_random_order.tsv"
FIRST_FRAME_DIR="/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/data/seadance2_yume_test_12classv2/first_frame"

# TSV="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/seadance2_yume_test_12classv3/world_model_action12_test_120_simple1cam_actionfirst.tsv"
# FIRST_FRAME_DIR="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/seadance2_yume_test_12classv3/first_frame"

# VLM backend: "internvl" (local InternVL3-2B, default) or "gpt" (Azure OpenAI API)
VLM_BACKEND="gpt"

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
#     --vlm_backend "${VLM_BACKEND}" \
#     --resume_from_checkpoint "${CKPT}"  #去掉就是加载原版

### ── Stage1 + LoRA ──────────────────────────────────────────────────────────
# LORA_CKPT="/mnt/bn/voyager-sg-l3/zhexiao.xiong/world_model/ckpts/YUME/outputs_yume1.5_stage1_lora_seadancev3/20260318_080459/lora-checkpoint-1000"
# LORA_RANK=128
# LORA_ALPHA=128
# OUTPUT_DIR_LORA="./outputs/stage1_lora_valv2_batch_3.18_gpt_1000steps"

# torchrun --nproc_per_node 8 --master_port 9603 \
#     fastvideo/sample/sample_5b_batch_val.py \
#     --seed 43 \
#     --gradient_checkpointing \
#     --train_batch_size=1 \
#     --max_sample_steps=600000 \
#     --mixed_precision="bf16" \
#     --allow_tf32 \
#     --video_output_dir="${OUTPUT_DIR_LORA}" \
#     --tsv_path="${TSV}" \
#     --first_frame_dir="${FIRST_FRAME_DIR}" \
#     --num_euler_timesteps 50 \
#     --rand_num_img 0.6 \
#     --skip_vlm \
#     --auto_caption \
#     --auto_caption_steps 6 \
#     --vlm_backend "${VLM_BACKEND}" \
#     --use_lora \
#     --lora_rank ${LORA_RANK} \
#     --lora_alpha ${LORA_ALPHA} \
#     --resume_from_checkpoint "${LORA_CKPT}"

### ── Stage 2/3 (full checkpoint, MVDT, 4 steps) ──────────────────────────
STAGE23_CKPT="/mnt/bn/voyager-sg-l3/zhexiao.xiong/world_model/outputs_train_yume1.5_stage23_seadancev3/20260318_231152/checkpoint-200"
OUTPUT_DIR_STAGE23="./outputs/stage23_val_batch_seadancev2_200steps_8steps"

torchrun --nproc_per_node 8 --master_port 9604 \
    fastvideo/sample/sample_5b_batch_val.py \
    --seed 43 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_sample_steps=600000 \
    --mixed_precision="bf16" \
    --allow_tf32 \
    --video_output_dir="${OUTPUT_DIR_STAGE23}" \
    --tsv_path="${TSV}" \
    --first_frame_dir="${FIRST_FRAME_DIR}" \
    --num_euler_timesteps 8 \
    --rand_num_img 0.6 \
    --skip_vlm \
    --auto_caption \
    --auto_caption_steps 6 \
    --vlm_backend "${VLM_BACKEND}" \
    --MVDT \
    --num_samples_per_class 10 \
    --resume_from_checkpoint "${STAGE23_CKPT}"

cd /mnt/bn/voyager-sg-l3/zhexiao.xiong
python runner.py
# --num_samples_per_class 5 \ # 每个类生成5个样本