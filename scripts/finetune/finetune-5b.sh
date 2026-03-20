#!/usr/bin/bash	

# DATA_DIR=./data
# IP=[MASTER NODE IP]
# gradient_checkpointing需要开，要不80G显存不够用
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="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"

WAN_CKPT="/mnt/bn/voyager-sg-l3/zhexiao.xiong/world_model/ckpts/Wan/Wan2.2-TI2V-5B"

## normal training (full finetune from Wan2.2-TI2V-5B) ######
torchrun --nproc_per_node 8 --master_port 9208 \
    train-5b.py \
    --seed 42 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=5000 \
    --learning_rate=5e-5 \
    --discriminator_learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps=200 \
    --checkpoints_total_limit=5 \
    --validation_steps 50 \
    --allow_tf32 \
    --t5_cpu \
    --ckpt_dir="${WAN_CKPT}" \
    --output_dir="ckpts/YUME/outputs_wan22-5b_stage1_seadancev3" \
    --sample_output_dir="outputs_train_wan22-5b_stage1_seadancev3" \
    --fps=16 \
    --root_dir="/mnt/bn/voyager-sg-l3/zhexiao.xiong/world_model/data/seadance2_yume_v3/mp4_frame" \
    --full_mp4="/mnt/bn/voyager-sg-l3/zhexiao.xiong/world_model/data/seadance2_yume_v3/Sekai/" \
    --use_wandb \
    --wandb_project="wan22-5B-stage1-seadance2"

### lora training (LoRA finetune from Wan2.2-TI2V-5B)
# torchrun --nproc_per_node 8 --master_port 9608 \
#     train-5b.py \
#     --seed 42 \
#     --gradient_checkpointing \
#     --train_batch_size=1 \
#     --dataloader_num_workers 16 \
#     --gradient_accumulation_steps=1 \
#     --max_train_steps=4000 \
#     --learning_rate=5e-5 \
#     --discriminator_learning_rate=5e-5 \
#     --mixed_precision="bf16" \
#     --checkpointing_steps=200 \
#     --checkpoints_total_limit=5 \
#     --validation_steps 50 \
#     --allow_tf32 \
#     --t5_cpu \
#     --ckpt_dir="${WAN_CKPT}" \
#     --output_dir="ckpts/YUME/outputs_wan22_stage1_lora_seadancev3" \
#     --sample_output_dir="outputs_train_wan22_stage1_lora_seadancev3" \
#     --fps=24 \
#     --root_dir="/mnt/bn/voyager-sg-l3/zhexiao.xiong/world_model/data/seadance2_yume_v3/mp4_frame" \
#     --full_mp4="/mnt/bn/voyager-sg-l3/zhexiao.xiong/world_model/data/seadance2_yume_v3/Sekai/" \
#     --use_wandb \
#     --wandb_project="wan22-stage1-lora-seadancev3" \
#     --wandb_run_name="wan22_stage1_lora_seadancev3" \
#     --use_lora \
#     --lora_alpha=128 \
#     --lora_rank=128
    
    # max train steps600000
    # validation steps 24

cd /mnt/bn/voyager-sg-l3/zhexiao.xiong
python runner.py