#!/usr/bin/bash	

# DATA_DIR=./data
# IP=[MASTER NODE IP]
export TOKENIZERS_PARALLELISM=false
### I2V
torchrun --nproc_per_node 8 --master_port 9600 \
    fastvideo/sample/sample_5b.py \
    --seed 43 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_sample_steps=600000 \
    --mixed_precision="bf16" \
    --allow_tf32 \
    --video_output_dir="./outputs/test_3.5_12class_grasp" \
    --caption_path="./caption_re.txt" \
    --test_data_dir="./val" \
    --num_euler_timesteps 50 \
    --rand_num_img 0.6 \
    --jpg_dir="/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/veo3_yume_training_12class/first_frame/grasp/" \
    --prompt "From a first-person perspective, grasp." \
    --resume_from_checkpoint "ckpts/YUME/outputs_yume1.5_stage1_3.4/20260305_003617/checkpoint-3600"  # <-- Stage1 checkpoint

# num_euler_timesteps  stage1 50 steps. stage2.3 4 steps.
### T2V
torchrun --nproc_per_node 8 --master_port 9600 \
    fastvideo/sample/sample_5b.py \
    --seed 43 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_sample_steps=600000 \
    --mixed_precision="bf16" \
    --allow_tf32 \
    --video_output_dir="./outputs/test" \
    --caption_path="./caption_re.txt" \
    --test_data_dir="./val" \
    --num_euler_timesteps 50 \
    --rand_num_img 0.6 \
    --T2V \
    --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage." \
    --resume_from_checkpoint "ckpts/YUME/outputs_yume1.5_stage1_3.4/20260305_003617/checkpoint-3600" 



    # --jpg_dir="./jpg/" \
    #--jpg_dir="./jpg/" \
    #--video_root_dir "./test_video"

### ── Stage 2/3 (full checkpoint, MVDT + Distil, 4 steps) ─────────────────
## I2V
# torchrun --nproc_per_node 8 --master_port 9600 \
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
#     --prompt "From a first-person perspective, grasp." \
#     --resume_from_checkpoint "outputs_train_yume1.5_stage23/TIMESTAMP/checkpoint-STEP"

## T2V
# torchrun --nproc_per_node 8 --master_port 9600 \
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
#     --resume_from_checkpoint "outputs_train_yume1.5_stage23/TIMESTAMP/checkpoint-STEP"
