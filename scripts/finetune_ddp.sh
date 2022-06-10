#!/bin/bash

ln -s /sharefs/baaihealth/xiaohongwang/hxiao/.conda/envs/torch1.8 /opt/conda/envs/
source activate
conda activate torch1.8
cd /sharefs/baaihealth/xiaohongwang/hxiao/codes/MAE_COVID19/
chmod -R 777 /sharefs/baaihealth/xiaohongwang/hxiao/codes/MAE_COVID19

#common params
batch_size=16
model='vit_large_patch16'
finetune='/sharefs/baaihealth/xiaohongwang/hxiao/codes/MAE_COVID19/output_finetune/C_orig_8:1:1_mae_pretrain_vit_large/checkpoint-best.pth'
# finetune='/sharefs/baaihealth/xiaohongwang/medical_pretrained_models/MAE/mae_pretrain_vit_base.pth'
# finetune='/sharefs/baaihealth/xiaohongwang/medical_pretrained_models/MAE/C_model_TFS_with_MAE_799.pth'
tar='U_sani'
split_ratio=[2:3:5]
# split_ratio=[8:1:1]
# resume='/sharefs/baaihealth/xiaohongwang/hxiao/codes/MAE_COVID19/output/C_orig_8:1:1_mae_pretrain_vit_base/checkpoint-250.pth'

python main_finetune_txt_ddp.py \
        --save_all \
        --batch_size ${batch_size} \
        --finetune ${finetune} \
        --model ${model} \
        --tar ${tar} \
        --split_ratio ${split_ratio} \
        --blr 5e-4 --layer_decay 0.65 \
        --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 main_finetune_txt_ddp.py \
#         --save_all \
#         --batch_size ${batch_size} \
#         --finetune ${finetune} \
#         --model ${model} \
#         --tar ${tar} \
#         --split_ratio ${split_ratio} 
#         # --resume ${resume}
        
 
chmod -R 777 /sharefs/baaihealth/xiaohongwang/hxiao/codes/MAE_COVID19

rlaunch --cpu=4 --gpu=1 --memory=8000 -- python main_finetune_txt_ddp.py --batch_size 16 --finetune /sharefs/baaihealth/xiaohongwang/MAE_COVID19/output_pretrain/CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_C_orig_L_orig_CC_orig_pretrain/checkpoint-300.pth --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --fft true