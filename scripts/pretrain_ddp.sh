#!/bin/bash

# ln -s /mnt/sfs_turbo/hxiao/.conda/envs/torch1.8 /opt/conda/envs/

# # for health
# mkdir /home/hwxiao/soft_link_health
# ln -s /home/hwxiao/soft_link_health /sharefs/baaihealth/xiaohongwang/mycodes/MAE_COVID19 
# ln -s /sharefs/baaihealth/xiaohongwang/MAE_COVID19_output /home/hwxiao/soft_link_health
# ln -s /sharefs/baaihealth/xiaohongwang/mycodes/MAE_COVID19/wandb  /home/hwxiao/soft_link_health

# # for share
# mkdir /home/hwxiao/soft_link_health
# ln -s /sharefs/healthshare/xiaohongwang/MAE_COVID19_output /home/hwxiao/soft_link_health


# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain_ddp.py \
#         --batch_size=64 \
#         --norm_pix_loss \
#         --blr 1.5e-4 --weight_decay 0.05 \
#         --mask_ratio 0.75 \
#         --epochs 800 \
#         --warmup_epochs 40 \
#         --dataset C1000 C_orig    

# torchrun --nnodes=1 --nproc_per_node=8 main_pretrain_ddp.py \
#          --batch_size=64 --norm_pix_loss --blr 1.5e-4 --weight_decay 0.05 --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 \
#          --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM CXC L_orig CC_orig 
#          --resume /sharefs/baaihealth/xiaohongwang/MAE_COVID19/output_pretrain/CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_C_orig_L_orig_CC_orig_pretrain/checkpoint-20.pth

OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=8 main_pretrain_ddp.py \
            --batch_size=64 \
            --norm_pix_loss \
            --blr 1.5e-4 \
            --weight_decay 0.05 \
            --mask_ratio 0.75 \
            --epochs 800 \
            --warmup_epochs 40 \
            --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM CXC L_orig CC_orig \
                      CXSD CRDX CDX QUEX CCXD MRC CHEXD \
                      CHAOSCT DL KITS LIDC LITS MMWHS