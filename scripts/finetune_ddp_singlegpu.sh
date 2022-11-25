#!/bin/bash

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

# OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 main_finetune_txt_ddp.py \
#         --save_all \
#         --batch_size ${batch_size} \
#         --finetune ${finetune} \
#         --model ${model} \
#         --tar ${tar} \
#         --split_ratio ${split_ratio} 
#         # --resume ${resume}

 
python main_finetune_txt_ddp.py --batch_size 16 --finetune /sharefs/baaihealth/xiaohongwang/MAE_COVID19/output_pretrain/CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_C_orig_L_orig_CC_orig_pretrain/checkpoint-300.pth --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --fft true
python main_finetune_txt_ddp.py --batch_size 16 --finetune /sharefs/baaihealth/xiaohongwang/MAE_COVID19/output_pretrain/base_CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_C_orig_L_orig_CC_orig_pretrain/checkpoint-300.pth --blr 0.005512 --cutmix 0.02317 --drop_path 0.06941 --layer_decay 0.606 --mixup 0.5743 --reprob 0.28  --weight_decay 0.7442  --fft true
python main_finetune_txt_ddp.py --batch_size 8 --finetune /sharefs/baaihealth/xiaohongwang/medical_pretrained_models/MAE/C_orig_8:1:1_mae_pretrain_vit_base.pth  --blr 0.0029628906306211466 --cutmix 0.2463813127293306 --drop_path 0.21969988402746932 --layer_decay 0.7373740569816005 --mixup 0.6614802880227085 --reprob 0.3167303518852447  --weight_decay 0.06957806900582497  --fft false --bias true --attn false --mlp false --block_list 0,1,2,3,4,5,6,7,8,9,10,11 --split_ratio 4.9:0.1:5 --tar U_orig
python main_finetune_txt_ddp_copy.py --batch_size 16 --finetune /sharefs/baaihealth/xiaohongwang/medical_pretrained_models/MAE/C_orig_8:1:1_mae_pretrain_vit_base.pth  --blr 0.0029628906306211466 --cutmix 0.2463813127293306 --drop_path 0.21969988402746932 --layer_decay 0.7373740569816005 --mixup 0.6614802880227085 --reprob 0.3167303518852447  --weight_decay 0.06957806900582497  --bias true --attn false --mlp false --reinit_blocks 1
python main_finetune_txt_ddp.py --batch_size 16 --finetune /sharefs/baaihealth/xiaohongwang/medical_pretrained_models/MAE/data21_mae_pretrain_vit_base.pth  --blr 0.00551190679498907 --cutmix 0.0231737608320578 --drop_path 0.06940750988462563 --layer_decay 0.6059904284863553 --mixup 0.574321765050360 --reprob 0.2799533432253266 --weight_decay 0.7442308839557386
python main_finetune_txt_ddp.py --batch_size 16 --finetune /sharefs/baaihealth/xiaohongwang/medical_pretrained_models/MAE/C_orig_8:1:1_mae_pretrain_vit_base.pth  --blr 0.0029628906306211466 --cutmix 0.2463813127293306 --drop_path 0.21969988402746932 --layer_decay 0.7373740569816005 --mixup 0.6614802880227085 --reprob 0.3167303518852447  --weight_decay 0.06957806900582497  --mlp true
python main_finetune_txt_ddp.py --batch_size 16 --finetune /sharefs/healthshare/xiaohongwang/MAE_COVID19_output/output_pretrain/base_CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_CXC_L_orig_CC_orig_CXSD_CRDX_CDX_QUEX_CCXD_MRC_CHEXD_CHAOSCT_DL_KITS_LIDC_LITS_MMWHS_DR_MNS_MURA_CXIU_OCX_pretrain/checkpoint-290.pth  --blr 0.0029628906306211466 --cutmix 0.2463813127293306 --drop_path 0.21969988402746932 --layer_decay 0.7373740569816005 --mixup 0.6614802880227085 --reprob 0.3167303518852447  --weight_decay 0.06957806900582497



python main_finetune_txt_ddp.py --model vit_base_patch16  --batch_size 64 --blr 0.0005 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --tar CXC --split_ratio 8:1:1 --finetune 
python main_finetune_txt_ddp.py --model vit_large_patch16  --batch_size 64 --blr 0.001 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --tar CXC --split_ratio 8:1:1 --finetune /home/hwxiao/medical_pretrained_models/MAE/data13_mae_pretrain_vit_large.pth

#--blr 0.005662050315933436 --cutmix 0.48801002523621617 --drop_path 0.03626496567656723 --layer_decay 0.7540216457239383 --mixup 0.55574615930789 --reprob 0.05080341932073407 --weight_decay 0.4492555267595793