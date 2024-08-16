import sys
sys.path.append('..')
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import models_mae
from torchvision import transforms

def show_image(image, title=''):
    # image is [H, W, 3]
    # assert image.shape[2] == 3
    plt.imshow(image)
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=12)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model):
    x = torch.tensor(img)
    x = x.unsqueeze(0)
    print(x.shape)
    
    # make it a batch-like
    # x = x.unsqueeze(dim=0)
    # print(x.shape)

    x = torch.einsum('nhwc->nchw', x)
    print(x.shape)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    print(y.shape)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    print(y.shape)

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [50, 50]
    plt.rcParams['axes.titlesize'] = 10 #子图的标题大小
    plt.figure(figsize=(15,15))
    plt.subplot(1, 4, 1)
    show_image(x[0], "a. Original image")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "b. Masked image")

    plt.subplot(1, 4, 3)
    show_image(y[0], "c. Reconstructed image")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "d. Reconstruction + visible patches")

    # plt.show()
    plt.savefig('result.png')

if __name__ == '__main__':    

    # root_dir = './data_dir/public_medical_images/selected4pretrain/COVID-CT/COVID-19-20_v2/data/nii2png'
    root_dir = './data_dir/public_medical_images/selected4pretrain/COVID-CT/COVID19-CT-Dataset1000+/data/dcm2png'
    # root_dir = './data_dir/public_medical_images/selected4finetune/UCSD_AI4H_COVID_CT_data/Images-processed/U_sani2'
    class_dirs = os.listdir(root_dir)
    mean, std = 0, 0
    count = 0

    for sub_dir in class_dirs:
        # for file_path in glob.glob(os.path.join(root_dir,sub_dir)):
        for file_path in glob.glob(os.path.join(root_dir, sub_dir, '*.png')):
            img = np.array(Image.open(file_path))
            img = transforms.functional.to_tensor(img)
            mean += torch.mean(img)
            std += torch.std(img)
            count += 1

    # print('mean: ', mean / count)
    # print('std: ', std / count)

    imagenet_mean = np.array([mean]) #0.3320 #0.6240
    imagenet_std = np.array([std]) 

    # imagenet_mean = np.array([0.3320]) #0.3320 #0.6240
    # imagenet_std = np.array([0.3341])  #0.3341 #0.3080

    img = Image.open('./data_dir/public_medical_images/selected4pretrain/COVID-CT/COVID19-CT-Dataset1000+/data/dcm2png/Subject (1)/Mediastinum_150.png').convert('RGB')
    # img = Image.open('./data_dir/public_medical_images/selected4pretrain/COVID-CT/COVID-19-20_v2/data/nii2png/volume-covid19-A-0003_116.png').convert('RGB')
    # img = Image.open('./data_dir/public_medical_images/selected4finetune/UCSD_AI4H_COVID_CT_data/Images-processed/U_sani2/COVID/kjr-21-e25-p1-10.png').convert('RGB')
    print(np.array(img).shape)
    # print(np.array(img))
    img = img.resize((224, 224))
    img = np.array(img)[:, :, :] / 255

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    print(img.shape)
    # plt.rcParams['figure.figsize'] = [5, 5]
    # show_image(torch.tensor(img))


    # chkpt_dir = './ckpts_dir/medical_pretrained_models/MAE/mae_pretrain_vit_base.pth'
    # chkpt_dir = './ckpts_dir/medical_pretrained_models/MAE/mae_finetuned_vit_base.pth'
    # chkpt_dir = './ckpts_dir/medical_pretrained_models/MAE/C_orig_8:1:1_mae_pretrain_vit_base.pth'
    # chkpt_dir = './ckpts_dir/medical_pretrained_models/MAE/C_orig_MAE_checkpoint_799.pth'
    # chkpt_dir = './ckpts_dir/medical_pretrained_models/MAE/C1000_C_orig_MAE_checkpoint-799.pth'
    # chkpt_dir = './ckpts_dir/medical_pretrained_models/MAE/C1000_MAE_checkpoint_799.pth'
    chkpt_dir = './MAE_COVID19_output_pretrain/base_CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_C_orig_L_orig_CC_orig_pretrain/checkpoint-799.pth'
    model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
    # chkpt_dir = './MAE_COVID19_output_pretrain/large_CHE_CCT_C1920_CAR_CCS_C1000_CIC_MRA_MRB_S_orig_SIRM_C_orig_L_orig_CC_orig_pretrain/checkpoint-799.pth'
    # model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')


    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(42)
    print('MAE with pixel reconstruction:')
    run_one_image(img, model_mae)