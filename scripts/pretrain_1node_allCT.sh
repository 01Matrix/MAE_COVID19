#!/bin/bash
# cd /home/hwxiao/mycodes/MAE_COVID19
# source activate /home/hwxiao/.conda/envs/pt12

# # for health
# mkdir /home/hwxiao/soft_link_health
# ln -s /home/hwxiao/soft_link_health /sharefs/baaihealth/xiaohongwang/mycodes/MAE_COVID19 
# ln -s /sharefs/baaihealth/xiaohongwang/MAE_COVID19_output /home/hwxiao/soft_link_health
# ln -s /sharefs/baaihealth/xiaohongwang/MAE_COVID19_output/wandb /home/hwxiao/soft_link_health

# # for share
# mkdir /home/hwxiao/soft_link_health
# ln -s /sharefs/healthshare/xiaohongwang/MAE_COVID19_output /home/hwxiao/soft_link_health
# ln -s /sharefs/healthshare/xiaohongwang/MAE_COVID19_output/wandb /home/hwxiao/soft_link_health

model_size=base
model_name=mae_vit_${model_size}_patch16

# CUDA_VISIBLE_DEVICES=4 python main_pretrain.py \
LOGLEVEL=INFO OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 main_pretrain.py \
                        --jobtype allCT \
                        --batch_size=128 \
                        --norm_pix_loss \
                        --blr 1.5e-4 \
                        --weight_decay 0.05 \
                        --mask_ratio 0.75 \
                        --epochs 800 \
                        --warmup_epochs 40 \
                        --model ${model_name} \
                        --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM CXC L_orig CC_orig \
                        CHAOSCT DL KITS LIDC LITS MMWHS VERSE LYMPH
                        # --resume ${resume}