#!/usr/bin/bash	

# DATA_DIR=./data
# IP=[MASTER NODE IP]
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai"


# train full sekai
# torchrun --nproc_per_node 8 --master_port 9607 \
#     fastvideo/distill_model.py \
#     --seed 42 \
#     --gradient_checkpointing \
#     --train_batch_size=1 \
#     --dataloader_num_workers 4 \
#     --gradient_accumulation_steps=1 \
#     --max_train_steps=600000 \
#     --learning_rate=1e-5 \
#     --discriminator_learning_rate=1e-5 \
#     --mixed_precision="bf16" \
#     --checkpointing_steps=100 \
#     --validation_steps 24 \
#     --allow_tf32 \
#     --MVDT \
#     --Distil \
#     --t5_cpu \
#     --ckpt_dir="./Yume-5B-720P" \
#     --root_dir="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/yume_training/mp4_frame" \
#     --full_mp4="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/yume_training/Sekai/" \
#     --output_dir="ckpts/YUME/outputs_train_yume1.5_stage23" \
#     --resume_from_checkpoint="ckpts/YUME/outputs_yume1.5/20260221_175548/checkpoint-3000"

# train self constructed data
torchrun --nproc_per_node 8 --master_port 9607 \
    fastvideo/distill_model.py \
    --seed 42 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000 \
    --learning_rate=1e-5 \
    --discriminator_learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps=200 \
    --validation_steps 50 \
    --allow_tf32 \
    --MVDT \
    --Distil \
    --t5_cpu \
    --ckpt_dir="./Yume-5B-720P" \
    --root_dir="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/seadance2_yume_v3/mp4_frame" \
    --full_mp4="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/seadance2_yume_v3//Sekai/" \
    --output_dir="outputs_train_yume1.5_stage23_seadancev3" \
    --resume_from_checkpoint="/mnt/bn/voyager-sg-l3/zhexiao.xiong/world_model/ckpts/YUME/outputs_yume1.5_stage1_lora_seadancev3/20260318_080459/lora-checkpoint-2000" \
    --use_wandb \
    --wandb_project="yume-stage23"
    # max train steps600000

cd /mnt/bn/voyager-sg-l3/zhexiao.xiong
python runner.py