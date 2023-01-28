#!/bin/bash

# # for health
# mkdir /home/hwxiao/soft_link_health
# ln -s /home/hwxiao/soft_link_health /sharefs/baaihealth/xiaohongwang/mycodes/MAE_COVID19 
# ln -s /sharefs/baaihealth/xiaohongwang/MAE_COVID19_output /home/hwxiao/soft_link_health
# ln -s /sharefs/baaihealth/xiaohongwang/MAE_COVID19_output/wandb /home/hwxiao/soft_link_health

# # for share
# mkdir /home/hwxiao/soft_link_health
# ln -s /sharefs/healthshare/xiaohongwang/MAE_COVID19_output /home/hwxiao/soft_link_health
# ln -s /sharefs/healthshare/xiaohongwang/MAE_COVID19_output/wandb /home/hwxiao/soft_link_health
MASTER_ADDR='172.24.71.50'

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_pretrain_ddp.py \
# OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 main_pretrain_ddp.py \
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 \
--master_addr=$MASTER_ADDR --master_port=25641 --use_env main_pretrain_ddp.py \
                        --jobtype data36 \
                        --batch_size=64 \
                        --norm_pix_loss \
                        --blr 1.5e-4 \
                        --weight_decay 0.05 \
                        --mask_ratio 0.75 \
                        --epochs 800 \
                        --warmup_epochs 40 \
                        --model mae_vit_base_patch16 \
                        --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM CXC L_orig CC_orig \
                                    CXSD CRDX CDX QUEX CCXD MRC CHEXD \
                                    CHAOSCT DL KITS LIDC LITS MMWHS VERSE LYMPH \
                                    CXNIH CXPERT DR MNS MURA CXIU OCX \
                        --resume /home/hwxiao/mycodes/MAE_COVID19/outputs/output_pretrain/base_CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_CXC_L_orig_CC_orig_CXSD_CRDX_CDX_QUEX_CCXD_MRC_CHEXD_CHAOSCT_DL_KITS_LIDC_LITS_MMWHS_VERSE_LYMPH_CXNIH_CXPERT_DR_MNS_MURA_CXIU_OCX_pretrain/checkpoint-280.pth
