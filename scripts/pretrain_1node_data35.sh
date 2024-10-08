#!/bin/bash

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 main_pretrain.py \
                        --jobtype data35 \
                        --batch_size=128 \
                        --norm_pix_loss \
                        --blr 1.5e-4 \
                        --weight_decay 0.05 \
                        --mask_ratio 0.75 \
                        --epochs 800 \
                        --warmup_epochs 40 \
                        --model mae_vit_large_patch16 \
                        --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM L_orig CC_orig \
                                    CXSD CRDX CDX QUEX CCXD MRC CHEXD \
                                    CHAOSCT DL KITS LIDC LITS MMWHS VERSE LYMPH \
                                    CXNIH CXPERT DR MNS MURA CXIU OCX \
                        --resume ./MAE_COVID19_output_pretrain/large_CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_L_orig_CC_orig_CXSD_CRDX_CDX_QUEX_CCXD_MRC_CHEXD_CHAOSCT_DL_KITS_LIDC_LITS_MMWHS_VERSE_LYMPH_CXNIH_CXPERT_DR_MNS_MURA_CXIU_OCX_pretrain/checkpoint-420.pth

