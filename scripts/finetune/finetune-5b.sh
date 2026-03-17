#!/usr/bin/bash	

# DATA_DIR=./data
# IP=[MASTER NODE IP]
# gradient_checkpointing需要开，要不80G显存不够用
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="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"

## normal training ######
# torchrun --nproc_per_node 8 --master_port 9208 \
#     train-5b.py \
#     --seed 42 \
#     --gradient_checkpointing \
#     --train_batch_size=1 \
#     --dataloader_num_workers 8 \
#     --gradient_accumulation_steps=1 \
#     --max_train_steps=10000 \
#     --learning_rate=1e-5 \
#     --discriminator_learning_rate=1e-5 \
#     --mixed_precision="bf16" \
#     --checkpointing_steps=200 \
#     --checkpoints_total_limit=5 \
#     --validation_steps 50 \
#     --allow_tf32 \
#     --t5_cpu \
#     --output_dir="ckpts/YUME/outputs_yume1.5_stage1_3.4" \
#     --sample_output_dir="outputs_train_yume1.5_stage1_3.4" \
#     --fps=16 \
#     --root_dir="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/veo3_yume_training_12class/mp4_frame" \
#     --full_mp4="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/veo3_yume_training_12class/Sekai/" \
#     --use_wandb \
#     --wandb_project="yume-stage1-veo3" \
#     # --wandb_run_name="my_run" \
#     # --use_lora \
#     # --lora_alpha=256 \
#     # --lora_rank=128 
    
#     # max train steps600000

### lora training
torchrun --nproc_per_node 8 --master_port 9608 \
    train-5b.py \
    --seed 42 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 8 \
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
    --output_dir="ckpts/YUME/outputs_yume1.5_stage1_lora_seadancev2" \
    --sample_output_dir="outputs_train_yume1.5_stage1_lora_seadancev2_3.14" \
    --fps=24 \
    --root_dir="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/seadance2_yume/mp4_frame" \
    --full_mp4="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/seadance2_yume/Sekai/" \
    --use_wandb \
    --wandb_project="yume-stage1-veo3-lora-seadancev2" \
    --wandb_run_name="yume1.5_stage1_lora-seadancev2" \
    --use_lora \
    --lora_alpha=128 \
    --lora_rank=128 
    
    # max train steps600000
    # validation steps 24