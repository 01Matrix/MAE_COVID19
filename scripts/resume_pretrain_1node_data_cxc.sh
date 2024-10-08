#!/bin/bash



# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 main_pretrain.py \
                        --jobtype cxc_resume_from_in1k \
                        --batch_size=128 \
                        --norm_pix_loss \
                        --blr 1.5e-4 \
                        --weight_decay 0.05 \
                        --mask_ratio 0.75 \
                        --epochs 800 \
                        --warmup_epochs 40 \
                        --model mae_vit_base_patch16 \
                        --dataset CXC \
                        --resume ./ckpts_dir/medical_pretrained_models/MAE/mae_pretrain_vit_base.pth