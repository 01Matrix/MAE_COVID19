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
MASTER_ADDR=172.24.65.4
model_size=large
model_name=mae_vit_${model_size}_patch16

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 \
# --master_addr=$MASTER_ADDR --master_port=25641 --use_env main_pretrain.py \
OMP_NUM_THREADS=1 torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$1 --master_addr=$MASTER_ADDR --master_port=25641 main_pretrain.py \
                        --jobtype data36 \
                        --batch_size=128 \
                        --norm_pix_loss \
                        --blr 1.5e-4 \
                        --weight_decay 0.05 \
                        --mask_ratio 0.75 \
                        --epochs 800 \
                        --warmup_epochs 40 \
                        --model ${model_name} \
                        --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM CXC L_orig CC_orig \
                                    CXSD CRDX CDX QUEX CCXD MRC CHEXD \
                                    CHAOSCT DL KITS LIDC LITS MMWHS VERSE LYMPH \
                                    CXNIH CXPERT DR MNS MURA CXIU OCX
