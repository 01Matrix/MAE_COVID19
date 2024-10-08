#!/bin/bash

batch_size=$1
model_size=$2
model=vit_${model_size}_patch16
tar=$3
split_ratio=$4
# finetune=./ckpts_dir/medical_pretrained_models/MAE/mae_pretrain_vit_${model_size}.pth
# finetune=./ckpts_dir/medical_pretrained_models/MAE/data13_mae_pretrain_vit_${model_size}.pth
# finetune=./ckpts_dir/medical_pretrained_models/MAE/data14_mae_pretrain_vit_${model_size}.pth
# finetune=./ckpts_dir/medical_pretrained_models/MAE/data21_mae_pretrain_vit_${model_size}.pth
# finetune=./ckpts_dir/medical_pretrained_models/MAE/data36_mae_pretrain_vit_${model_size}.pth
# finetune=./ckpts_dir/medical_pretrained_models/MAE/allCT_mae_pretrain_vit_${model_size}.pth
# finetune=./ckpts_dir/medical_pretrained_models/MAE/CXC_mae_pretrain_vit_${model_size}.pth

finetune=./ckpts_dir/medical_pretrained_models/inter_model/inter_CXC_8:1:1_mae_pretrain_vit_${model_size}.pth
# finetune=./ckpts_dir/medical_pretrained_models/inter_model/inter_CXC_8:1:1_data13_mae_vit_${model_size}.pth
# finetune=./ckpts_dir/medical_pretrained_models/inter_model/inter_CXC_8:1:1_data36_mae_vit_${model_size}.pth

# finetune=./ckpts_dir/medical_pretrained_models/TFS_model/CXC_8:1:1_vit_b_16.pth
# finetune=./ckpts_dir/medical_pretrained_models/TFS_model/CXC_8:1:1_vit_l_16.pth

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=$5 main_finetune.py \
        --model ${model} \
        --save_all \
        --batch_size ${batch_size} \
        --tar ${tar} \
        --split_ratio ${split_ratio} \
        --epochs 200 \
        --finetune ${finetune} \
        --ckpt_save_freq 1 \
        --blr 0.0029628906306211466 --cutmix 0.2463813127293306 --drop_path 0.21969988402746932 --layer_decay 0.7373740569816005 --mixup 0.6614802880227085 --reprob 0.3167303518852447  --weight_decay 0.06957806900582497
        # --blr 0.001 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
        # --resume ./MAE_COVID19_output_finetune/finetune_with_data36_mae_pretrain_vit_large/8:1:1/CXC_seed42_bs64_b0.001_l0.75_w0.05_d0.1_r0.25_m0.8_c1.0_rb0_fb0_attnFalse_mlpFalse_biasFalse_FFT/checkpoint-74.pth
  
# base    --blr 0.0005 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0

# large   --blr 0.001 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0

# base   --blr 0.0029628906306211466 --cutmix 0.2463813127293306 --drop_path 0.21969988402746932 --layer_decay 0.7373740569816005 --mixup 0.6614802880227085 --reprob 0.3167303518852447  --weight_decay 0.06957806900582497  --mlp true

# large  --blr 0.005662050315933436 --cutmix 0.48801002523621617 --drop_path 0.03626496567656723 --layer_decay 0.7540216457239383 --mixup 0.55574615930789 --reprob 0.05080341932073407 --weight_decay 0.4492555267595793