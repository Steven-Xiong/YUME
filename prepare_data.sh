
# cd vipe
# source /mnt/bn/voyager-sg-l3/zhexiao.xiong/miniconda3/bin/activate vipe
# bash run.sh
# cd /mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME

# transform vipe results to yume format
python scripts/prepare_vipe_to_yume.py \
    --vipe_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/data/seadance2_yume/12class_ego_anno \
    --rgb_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/data/seadance2_yume/video \
    --output_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/YUME/data/seadance2_yume \
    --clip_length 33 \
    --clip_stride 16

    # 33