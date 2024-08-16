#!/bin/bash



# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 main_pretrain.py \
                        --jobtype chexpert \
                        --batch_size=128 \
                        --norm_pix_loss \
                        --blr 1.5e-4 \
                        --weight_decay 0.05 \
                        --mask_ratio 0.75 \
                        --epochs 800 \
                        --warmup_epochs 40 \
                        --model mae_vit_large_patch16 \
                        --dataset CXPERT \
                        --resume /sharefs/healthshare/xiaohongwang/MAE_COVID19_output/output_pretrain/large_CXPERT_pretrain/checkpoint-150.pth