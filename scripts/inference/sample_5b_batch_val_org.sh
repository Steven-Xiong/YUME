#!/usr/bin/bash

export TOKENIZERS_PARALLELISM=false

# TSV="/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/data/seadance2_yume_test_12classv2/world_model_action12_test_240_random_order.tsv"
# FIRST_FRAME_DIR="/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/data/seadance2_yume_test_12classv2/first_frame"

TSV="/mnt/bn/voyager-sg-l3/zhexiao.xiong/world_model/data/seadance2_yume_test_12classv3/world_model_action12_test_120_simple1cam_actionfirst.tsv"
FIRST_FRAME_DIR="/mnt/bn/voyager-sg-l3/zhexiao.xiong/world_model/data/seadance2_yume_test_12classv3/first_frame"

# VLM backend: "internvl" (local InternVL3-2B, default) or "gpt" (Azure OpenAI API)
VLM_BACKEND="gpt"

### ── YUME 1.5 原版 checkpoint (no fine-tuning) ───────────────────────────
OUTPUT_DIR_ORIG="./outputs/yume1.5_orig_val_batch_seadancev3_4steps"

torchrun --nproc_per_node 8 --master_port 9605 \
    fastvideo/sample/sample_5b_batch_val.py \
    --seed 43 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_sample_steps=600000 \
    --mixed_precision="bf16" \
    --allow_tf32 \
    --video_output_dir="${OUTPUT_DIR_ORIG}" \
    --tsv_path="${TSV}" \
    --first_frame_dir="${FIRST_FRAME_DIR}" \
    --num_euler_timesteps 4 \
    --rand_num_img 0.6 \
    --skip_vlm \
    --auto_caption \
    --auto_caption_steps 10 \
    --num_samples_per_class 10 \
    --vlm_backend "${VLM_BACKEND}"
    # 不传 --resume_from_checkpoint，直接加载 ./Yume-5B-720P 原版权重

cd /mnt/bn/voyager-sg-l3/zhexiao.xiong
python runner.py
