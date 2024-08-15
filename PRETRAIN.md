## Pre-training MAE_COVID19

To pre-train ViT-Base with **one-node training**, run the following on 1 node with 8 GPUs:
```
#!/bin/bash

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 main_pretrain.py \
                        --jobtype data14 \
                        --batch_size=64 \
                        --norm_pix_loss \
                        --blr 1.5e-4 \
                        --weight_decay 0.05 \
                        --mask_ratio 0.75 \
                        --epochs 800 \
                        --warmup_epochs 40 \
                        --model mae_vit_base_patch16 \
                        --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM CXC L_orig CC_orig
```
- As for the `--dataset`, you can set various subsets, like `CHE CCT C1920` (seperated by white space), etc. In total, we have 36 subsets, including CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM CXC L_orig CC_orig CXSD CRDX CDX QUEX CCXD MRC CHEXD CHAOSCT DL KITS LIDC LITS MMWHS VERSE LYMPH CXNIH CXPERT DR MNS MURA CXIU OCX. If you select 14 subset, then set the `--jobtype` to `data14`.
- Here the effective batch size is 64 (`batch_size` per gpu) * 8 (`nodes`) * 8 (gpus per node) = 4096. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is `batch_size` (per gpu) * `nodes` * 8 (gpus per node) * `accum_iter`.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- Here we adopt `--norm_pix_loss` as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off `--norm_pix_loss`.

To train ViT-Large, set `--model mae_vit_large_patch16`.

To start training:
```
cd MAE_COVID19
bash scripts/pretrain_1node_data14.sh # use 14 COVID-19 CT subsets
```
Similarly, you can pretrain with various dataset composite:
```
cd MAE_COVID19
bash scripts/pretrain_1node_data_7xray.sh # use 7 2D X-ray subsets
bash scripts/pretrain_1node_data_allCT.sh # use all CT subsets
bash scripts/pretrain_1node_data_data36.sh # use all 36 subsets
```
