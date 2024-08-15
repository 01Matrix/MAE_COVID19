## Fine-tuning Pre-trained MAE for Medical Image Classification

### Evaluation

### Fine-tuning

Get our pre-trained checkpoints from [here](README.md#pre-training-recipes).

To fine-tune the pre-trained ViT-Base with **single-node training**, run the following on 1 node with 8 GPUs:
```
#!/bin/bash

batch_size=$1
model_size=$2
model=vit_${model_size}_patch16
tar=$3
split_ratio=$4

# ckpt=/home/hwxiao/medical_pretrained_models/MAE/mae_pretrain_vit_${model_size}.pth
# ckpt=/home/hwxiao/medical_pretrained_models/MAE/data13_mae_pretrain_vit_${model_size}.pth
# ckpt=/home/hwxiao/medical_pretrained_models/MAE/data14_mae_pretrain_vit_${model_size}.pth
# ckpt=/home/hwxiao/medical_pretrained_models/MAE/data21_mae_pretrain_vit_${model_size}.pth
# ckpt=/home/hwxiao/medical_pretrained_models/MAE/data36_mae_pretrain_vit_${model_size}.pth
# ckpt=/home/hwxiao/medical_pretrained_models/MAE/allCT_mae_pretrain_vit_${model_size}.pth
# ckpt=/home/hwxiao/medical_pretrained_models/MAE/CXC_mae_pretrain_vit_${model_size}.pth

ckpt=/home/hwxiao/medical_pretrained_models/inter_model/inter_CXC_8:1:1_mae_pretrain_vit_${model_size}.pth
# ckpt=/home/hwxiao/medical_pretrained_models/inter_model/inter_CXC_8:1:1_data13_mae_vit_${model_size}.pth
# ckpt=/home/hwxiao/medical_pretrained_models/inter_model/inter_CXC_8:1:1_data36_mae_vit_${model_size}.pth

# ckpt=/home/hwxiao/medical_pretrained_models/TFS_model/CXC_8:1:1_vit_b_16.pth
# ckpt=/home/hwxiao/medical_pretrained_models/TFS_model/CXC_8:1:1_vit_l_16.pth

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=$5 main_finetune.py \
        --model ${model} \
        --save_all \
        --batch_size ${batch_size} \
        --tar ${tar} \
        --split_ratio ${split_ratio} \
        --epochs 200 \
        --finetune ${ckpt} \
        --ckpt_save_freq 1 \
        --blr 0.001 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0
```
- Here the effective batch size is 32 (`batch_size` per gpu) * 4 (`nodes`) * 8 (gpus per node) = 1024.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- The hyper-parameters we adopt:
```
# vit_base    --blr 0.0005 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0

# vit_large   --blr 0.001 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0

# vit_base   --blr 0.0029628906306211466 --cutmix 0.2463813127293306 --drop_path 0.21969988402746932 --layer_decay 0.7373740569816005 --mixup 0.6614802880227085 --reprob 0.3167303518852447  --weight_decay 0.06957806900582497

# vit_large  --blr 0.005662050315933436 --cutmix 0.48801002523621617 --drop_path 0.03626496567656723 --layer_decay 0.7540216457239383 --mixup 0.55574615930789 --reprob 0.05080341932073407 --weight_decay 0.4492555267595793
```

To start training:
```
cd MAE_COVID19
bash scripts/finetune.sh 8 base U_orig 8:1:1 8
```
- batch_size=8
- model_size=base
- tar=U_orig  # the target downstream dataset
- split_ratio=8:1:1
- nproc_per_node=8


#### Notes

- The [pre-trained models](README.md#pre-training-recipes) are trained with *normalized* pixels `--norm_pix_loss` (1600 epochs, Table 3 in paper). The fine-tuning hyper-parameters are slightly different from the default baseline using *unnormalized* pixels.

- Here we use RandErase following DeiT: `--reprob 0.25`. Its effect is smaller than random variance.
