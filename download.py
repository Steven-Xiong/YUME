from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stdstu123/Yume-I2V-540P",
    local_dir="/mnt/bn/icvg/zhexiao.xiong/ckpts/YUME/Yume-I2V-540P",
    local_dir_use_symlinks=False
)