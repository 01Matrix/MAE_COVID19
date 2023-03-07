#!/bin/bash

batch_size=16
model_size=large
model=vit_${model_size}_patch16
tar=U_orig
split_ratio=2:3:5
# finetune=/home/hwxiao/medical_pretrained_models/MAE/mae_pretrain_vit_${model_size}.pth
# finetune=/home/hwxiao/medical_pretrained_models/MAE/data13_mae_pretrain_vit_${model_size}.pth
# finetune=/home/hwxiao/medical_pretrained_models/MAE/data14_mae_pretrain_vit_${model_size}.pth
# finetune=/home/hwxiao/medical_pretrained_models/MAE/data21_mae_pretrain_vit_${model_size}.pth
finetune=/home/hwxiao/medical_pretrained_models/MAE/data36_mae_pretrain_vit_${model_size}.pth
# finetune=/home/hwxiao/medical_pretrained_models/MAE/allCT_mae_pretrain_vit_${model_size}.pth
# finetune=/home/hwxiao/medical_pretrained_models/MAE/CXC_mae_pretrain_vit_${model_size}.pth

# finetune=/home/hwxiao/medical_pretrained_models/inter_model/inter_CXC_8:1:1_mae_pretrain_vit_${model_size}.pth
# finetune=/home/hwxiao/medical_pretrained_models/inter_model/inter_CXC_8:1:1_data13_mae_vit_${model_size}.pth
# finetune=/home/hwxiao/medical_pretrained_models/inter_model/inter_CXC_8:1:1_data36_mae_vit_${model_size}.pth

# finetune=/home/hwxiao/medical_pretrained_models/TFS_model/CXC_8:1:1_vit_b_16.pth
# finetune=/home/hwxiao/medical_pretrained_models/TFS_model/CXC_8:1:1_vit_l_16.pth

# OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 main_finetune.py \
python main_finetune.py \
        --model ${model} \
        --save_all \
        --batch_size ${batch_size} \
        --tar ${tar} \
        --split_ratio ${split_ratio} \
        --epochs 200 \
        --finetune ${finetune} \
        --blr 0.001 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0
        # --blr 0.0029628906306211466 --cutmix 0.2463813127293306 --drop_path 0.21969988402746932 --layer_decay 0.7373740569816005 --mixup 0.6614802880227085 --reprob 0.3167303518852447  --weight_decay 0.06957806900582497
  
# base    --blr 0.0005 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0

# large   --blr 0.001 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0

# base   --blr 0.0029628906306211466 --cutmix 0.2463813127293306 --drop_path 0.21969988402746932 --layer_decay 0.7373740569816005 --mixup 0.6614802880227085 --reprob 0.3167303518852447  --weight_decay 0.06957806900582497  --mlp true

# large  --blr 0.005662050315933436 --cutmix 0.48801002523621617 --drop_path 0.03626496567656723 --layer_decay 0.7540216457239383 --mixup 0.55574615930789 --reprob 0.05080341932073407 --weight_decay 0.4492555267595793