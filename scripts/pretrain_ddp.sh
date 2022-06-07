#!/bin/bash

ln -s /mnt/sfs_turbo/hxiao/.conda/envs/torch1.8 /opt/conda/envs/
source activate
conda activate torch1.8
cd /mnt/sfs_turbo/hxiao/codes/MAE_COVID19
chmod -R 777 /mnt/sfs_turbo/hxiao/codes/MAE_COVID19

# python /mnt/sfs_turbo/hxiao/codes/MAE_COVID19/main_pretrain_txt_ddp.py \
#         --batch_size 8 \
#         --output_dir '/mnt/sfs_turbo/hxiao/codes/MAE_COVID19/output' \
#         --dataset SI_orig
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 /mnt/sfs_turbo/hxiao/codes/MAE_COVID19/main_pretrain_txt_ddp.py \
#         --batch_size 16 --dataset L_orig

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 /mnt/sfs_turbo/hxiao/codes/MAE_COVID19/main_pretrain_ddp.py \
        --batch_size=64 \
        --norm_pix_loss \
        --blr 1.5e-4 --weight_decay 0.05 \
        --mask_ratio 0.75 \
        --epochs 800 \
        --warmup_epochs 40 \
        --dataset C1000 C_orig 

chmod -R 777 /mnt/sfs_turbo/hxiao/codes/MAE_COVID19     


