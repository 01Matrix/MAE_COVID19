#!/bin/bash

ln -s /mnt/sfs_turbo/hxiao/.conda/envs/torch1.8 /opt/conda/envs/
source activate
conda activate torch1.8
cd /mnt/sfs_turbo/hxiao/codes/MAE_COVID19/
chmod -R 777 /mnt/sfs_turbo/hxiao/codes/MAE_COVID19

# python main_finetune_ddp.py --finetune \
# /mnt/sfs_turbo/medical_pretrained_models/MAE/mae_finetuned_vit_base.pth --tar U_sani2 --seed 42 \
# --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0

finetune='/mnt/sfs_turbo/medical_pretrained_models/MAE/mae_finetuned_vit_base.pth'
# finetune='/mnt/sfs_turbo/medical_pretrained_models/MAE/mae_pretrain_vit_base.pth'
# finetune='/mnt/sfs_turbo/medical_pretrained_models/MAE/C1000_pretrain_checkpoint_799.pth'
model='vit_base_patch16'
batch_size=16
dataset=U_sani2
seed=1
# blr=5e-4 
# layer_decay=0.65
weight_decay=0.05
# drop_path=0.1 
# reprob=0.25 
# mixup=0.8
# cutmix=1.0 

for blr in 1e-3 5e-4
do
    for layer_decay in 0.65 0.75
    do
        for drop_path in 0.1 0.3
        do
            for reprob in 0.25 0.75
            do
                for mixup in 0 0.5 0.8
                do
                    for cutmix in 0 0.5 1.0
                    do
                        python main_finetune_ddp.py \
                                --batch_size $batch_size \
                                --finetune $finetune \
                                --model $model \
                                --tar $dataset \
                                --seed $seed \
                                --blr $blr \
                                --layer_decay $layer_decay \
                                --weight_decay $weight_decay \
                                --drop_path $drop_path \
                                --reprob $reprob \
                                --mixup $mixup \
                                --cutmix $cutmix 
                    done
                done
            done   
        done
    done
done
