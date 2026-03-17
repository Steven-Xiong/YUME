#!/usr/bin/bash	

# DATA_DIR=./data
# IP=[MASTER NODE IP]
# export TOKENIZERS_PARALLELISM=false
# ### I2V
# torchrun --nproc_per_node 8 --master_port 9600 \
#     fastvideo/sample/sample_5b.py \
#     --seed 43 \
#     --gradient_checkpointing \
#     --train_batch_size=1 \
#     --max_sample_steps=600000 \
#     --mixed_precision="bf16" \
#     --allow_tf32 \
#     --video_output_dir="./outputs/self1" \
#     --caption_path="./caption_self.txt" \
#     --test_data_dir="./val" \
#     --num_euler_timesteps 8 \
#     --rand_num_img 0.6 \
#     --jpg_dir="./data/action/stonehenge/" \
#     --prompt "Photo of Stonehenge on Salisbury Plain under low overcast sky, \
# light morning mist, soft diffuse lighting, ancient stones in calm, \
# wide landscape, high detail, 4K style stock photography."
### T2V
# torchrun --nproc_per_node 8 --master_port 9600 \
#     fastvideo/sample/sample_5b.py \
#     --seed 43 \
#     --gradient_checkpointing \
#     --train_batch_size=1 \
#     --max_sample_steps=600000 \
#     --mixed_precision="bf16" \
#     --allow_tf32 \
#     --video_output_dir="./outputs" \
#     --caption_path="./caption_re.txt" \
#     --test_data_dir="./val" \
#     --num_euler_timesteps 4 \
#     --rand_num_img 0.6 \
#     --T2V \
#     --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage." \


# batch infer 

# i2v
# bash infer_batch.sh --mode i2v --input_file batch_inputs.example.tsv --caption_path ./my_i2v_action.txt

# t2v
bash infer_batch.sh --mode t2v --input_file data/action_t2v/interaction_prompts_25.tsv --caption_path caption_action_self.txt