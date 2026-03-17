# python fastvideo/sample/visualize_batch_results.py \
#     --video_dir "./outputs/stage1_lora_val_batch_3.9_1200steps" \
#     --tsv_path "/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/veo3_yume_test_12class/world_model_action12_inference_240.tsv" \
#     --first_frame_dir "/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/veo3_yume_test_12class/first_frame" \
#     --output "./outputs/stage1_lora_val_batch_3.9_1200steps/results/results_lora1200steps.html"


python fastvideo/sample/visualize_batch_results.py \
    --video_dir "/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/outputs/stage1_lora_val_batch_still1000step_3.8_400" \
    --tsv_path "/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/veo3_yume_test_12class/world_model_action12_inference_240.tsv" \
    --first_frame_dir "/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/veo3_yume_test_12class/first_frame" \
    --output "/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/outputs/stage1_lora_val_batch_still1000step_3.8_400/results/results_lora400steps.html"


python fastvideo/sample/visualize_batch_results.py \
    --video_dir "/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/outputs/stage1_val_batch200step_autocaption_3.9_org" \
    --tsv_path "/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/veo3_yume_test_12class/world_model_action12_inference_240.tsv" \
    --first_frame_dir "/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/veo3_yume_test_12class/first_frame" \
    --output "/mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/outputs/stage1_val_batch200step_autocaption_3.9_org/results/results_org.html"