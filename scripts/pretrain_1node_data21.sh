#!/bin/bash

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
# use 21 datasets of COVID19: CT and X-ray
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 main_pretrain.py \
                        --jobtype data21_allCOVID \
                        --batch_size=64 \
                        --norm_pix_loss \
                        --blr 1.5e-4 \
                        --weight_decay 0.05 \
                        --mask_ratio 0.75 \
                        --epochs 800 \
                        --warmup_epochs 40 \
                        --model mae_vit_large_patch16 \
                        --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM CXC L_orig CC_orig \
                                    CXSD CRDX CDX QUEX CCXD MRC CHEXD

