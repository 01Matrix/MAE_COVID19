#!/bin/bash

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 \
        main_finetune_txt_ddp_multigpus.py \
        --model vit_large_patch16  --batch_size 64 --blr 0.001 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --tar CXC \
        --split_ratio 8:1:1 --finetune /home/hwxiao/mycodes/MAE_COVID19/outputs/output_pretrain/large_CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_L_orig_CC_orig_CXSD_CRDX_CDX_QUEX_CCXD_MRC_CHEXD_CHAOSCT_DL_KITS_LIDC_LITS_MMWHS_VERSE_LYMPH_CXNIH_CXPERT_DR_MNS_MURA_CXIU_OCX_pretrain/checkpoint-420.pth --seed 3407