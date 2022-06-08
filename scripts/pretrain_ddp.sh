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


# rlaunch --cpu=48 --gpu=8 --memory=200000 -- torchrun --nnodes=1 --nproc_per_node=8 main_pretrain_ddp.py --batch_size=64 --norm_pix_loss --blr 1.5e-4 --weight_decay 0.05 --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM C_orig L_orig CC_orig --resume /sharefs/baaihealth/xiaohongwang/MAE_COVID19/output_pretrain/CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_C_orig_L_orig_CC_orig_pretrain/checkpoint-20.pth