#!/usr/bin/bash	

# 快速验证脚本（train-5b.py）
# 适用于调试和测试

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node 8 --master_port 9608 \
    train-5b.py \
    --seed 42 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=100 \
    --learning_rate=1e-5 \
    --discriminator_learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps=25 \
    --validation_steps 24 \
    --allow_tf32 \
    --t5_cpu \
    --output_dir="./outputs_quick1.5"
