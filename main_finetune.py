# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
from curses import flash
import datetime
from distutils.command.config import config
import json
import numpy as np
import random
import wandb
from loguru import logger
import os
import time
from pathlib import Path
import torch
from torch import nn
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import util.data_loader_COVID19 as data_loader_COVID19
import timm
assert timm.__version__ == "0.5.4" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit
from engine_finetune import train_one_epoch, evaluate
# import fastfood



def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--ckpt_save_freq', type=int, default=10)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params, DeiT数据增强方法
    # 训练模型时，随机选取一个图片的矩形区域，将这个矩形区域的像素值用随机值或者平均像素值代替，产生局部遮挡的效果。该数据增强可以与随机切除、随机翻转等数据增强结合起来使用。
    # 在ReID、图像分类领域可以作为升点trick。
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--save_all', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data_dir/public_medical_images', type=str,
                        help='dataset path')
    parser.add_argument('--split_ratio',default='2:3:5',type=str,help='Split dataset to train:val:test')
    parser.add_argument('--tar', type=str, default='U_orig', help='finetune data')

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./MAE_COVID19_output_finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--in_chans', default=3, type=int,
                        help='input channel')
    
    # parser.add_argument('--eval', action='store_true',
    #                     help='Perform evaluation only')
    parser.add_argument('--test', action='store_true',
                        help='Perform test only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # partial fine-tuning parameters
    parser.add_argument('-fb','--frozen_blocks',type=int, default=0, help='number of frozen blocks')
    parser.add_argument('-rb','--reinit_blocks',type=int, default=0, help='number of blocks with re-init')
    parser.add_argument('--attn',type=str2bool, default=False,help='It means just finetune attention layer.')
    parser.add_argument('--mlp',type=str2bool, default=False,help='It means just finetune mlp layer')
    parser.add_argument('--bias',type=str2bool, default=False,help='It means just finetune bias term')
    parser.add_argument('--norm1',type=str2bool, default=False,help='It means just finetune norm1 parts')
    parser.add_argument('--norm2',type=str2bool, default=False,help='It means just finetune norm2 parts')
    # nargs=?，如果没有在命令行中出现对应的项，则给对应的项赋值为default。
    # 特殊的是，对于可选项，如果命令行中出现了此可选项，但是之后没有跟随赋值参数，则此时给此可选项并不是赋值default的值，而是赋值const的值。
    return parser

def str2bool(v):
    if isinstance(v, bool):
        print('str2bool')
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.use_deterministic_algorithms(True) #相当于自检，只要使用了不可复现的运算操作，代码就会自动报错。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

GLOBAL_SEED = 1 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

# Partially finetuning ATTENTION or MLP or norm parts or BIAS part.
def partial_ft(flag_attn,flag_mlp,flag_norm1,flag_norm2,flag_bias,model):

    for n, p in model.named_parameters():
        p.requires_grad = False

    # model.cls_token.requires_grad = True
    # model.pos_embed.requires_grad = True
    # for _, p in model.patch_embed.named_parameters():
    #     p.requires_grad = True

    #复制某一层的权重
    # model.state_dict()["model.blocks[0].mlp.fc1.weight"].copy_(model.state_dict()["model.blocks[1].mlp.fc1.weight"]) 

    if flag_attn:
        print('AND Finetune ATTENTION part.')
        for block in model.blocks[-1:]:
            for n, p in block.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
    if flag_mlp:
        print('AND Finetune MLP part of the last block.')
        for block in model.blocks[-1:]:
            for n, p in block.named_parameters():
                if 'mlp' in n:
                    p.requires_grad = True
    if flag_norm1:
        print('AND Finetune norm1 parts.')
        for block in model.blocks[-1:]:
            for n, p in block.named_parameters():
                if 'norm1' in n:
                    p.requires_grad = True
    if flag_norm2:
        print('AND Finetune norm2 parts.')
        for block in model.blocks[-1:]:
            for n, p in block.named_parameters():
                if 'norm2' in n:
                    p.requires_grad = True
    if flag_bias:
        print('AND Finetune BIAS terms.')
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True

    if hasattr(model, 'norm'):
        for _, p in model.norm.named_parameters():
            p.requires_grad = True
    if hasattr(model, 'fc_norm'):
        for _, p in model.fc_norm.named_parameters():
            p.requires_grad = True
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    if args.reinit_blocks > 0:
        reinit_parts(args.reinit_blocks,flag_attn,flag_mlp,flag_norm1,flag_norm2,flag_bias,model)
    
    for n, p in model.named_parameters():
        print(n,p.requires_grad)

def reinit_parts(reinit_blocks,flag_attn,flag_mlp,flag_norm1,flag_norm2,flag_bias,model):
    if flag_attn and reinit_blocks > 0:
        print(f' AND Reinitializing Last {reinit_blocks} Blocks\' ATTENTION parts')
        for block in model.blocks[-reinit_blocks:]:
            trunc_normal_(block.attn.qkv.weight,std=2e-5)
            trunc_normal_(block.attn.qkv.bias,std=1e-6)
            trunc_normal_(block.attn.proj.weight,std=2e-5)
            trunc_normal_(block.attn.proj.bias,std=1e-6)
    elif flag_mlp and reinit_blocks > 0:
        print(f'AND Reinitializing Last {reinit_blocks} Blocks\' MLP parts')
        for block in model.blocks[-reinit_blocks:]:
            trunc_normal_(block.mlp.fc1.weight,std=2e-5)
            trunc_normal_(block.mlp.fc1.bias,std=1e-6)
            trunc_normal_(block.mlp.fc2.weight,std=2e-5)
            trunc_normal_(block.mlp.fc2.bias,std=1e-6)
    elif flag_norm1 and reinit_blocks > 0:
        print(f'AND Reinitializing Last {reinit_blocks} Blocks\' norm1 parts')
        for block in model.blocks[-reinit_blocks:]:
            trunc_normal_(block.norm1.weight,std=2e-5)
            trunc_normal_(block.norm1.bias,std=1e-6)
    elif flag_norm2 and reinit_blocks > 0:
        print(f'AND Reinitializing Last {reinit_blocks} Blocks\' norm2 parts')
        for block in model.blocks[-reinit_blocks:]:
            trunc_normal_(block.norm2.weight,std=2e-5)
            trunc_normal_(block.norm2.bias,std=1e-6)
    elif flag_bias and reinit_blocks > 0:
        print(f'AND Reinitializing Last {reinit_blocks} Blocks\' BIAS parts')
        for block in model.blocks[-reinit_blocks:]:
            trunc_normal_(block.norm1.bias,std=1e-6)
            trunc_normal_(block.attn.qkv.bias,std=1e-6)
            trunc_normal_(block.attn.proj.bias,std=1e-6)
            trunc_normal_(block.norm2.bias,std=1e-6)
            trunc_normal_(block.mlp.fc1.bias,std=1e-6)
            trunc_normal_(block.mlp.fc2.bias,std=1e-6)

def reinit_blocks(reinit_blocks,model):
    if reinit_blocks > 0:
        print(f'Reinitializing Last {reinit_blocks} Blocks ...')
        for block in model.blocks[-reinit_blocks:]:
            # print(block.modules)
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    # module.weight.data.normal_(mean=0.0, std=2e-5)
                    trunc_normal_(module.weight, std=2e-5)
                    if module.bias is not None:
                        # module.bias.data.zero_()
                        trunc_normal_(module.bias, std=1e-6)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
        print('Re-initialization Done.!')
        # for sup_module in model.modules():
        #     for name, module in sup_module.named_children():
        #         print(name,module)
    # import pdb; pdb.set_trace()

def freeze_blocks(frozen_blocks,model):
    model.cls_token.requires_grad = False
    model.pos_embed.requires_grad = False
    for _, p in model.patch_embed.named_parameters():
        p.requires_grad = False
    # freeze an increasing number of lower blocks
    print(f'Freezing lower {frozen_blocks} Blocks ...')
    if frozen_blocks > 0:
        for block in model.blocks[:frozen_blocks]:
            for module in block.modules():
                for _, p in module.named_parameters():
                    p.requires_grad = False 

    if hasattr(model, 'norm'):
        for _, p in model.norm.named_parameters():
            p.requires_grad = True
    if hasattr(model, 'fc_norm'):
        for _, p in model.fc_norm.named_parameters():
            p.requires_grad = True
    for _,p in model.head.named_parameters():
        p.requires_grad = True

def main(args):

    misc.init_distributed_mode(args)
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print('log dir:{}'.format(os.path.join(args.output_dir,args.save_dir)))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    print('actual seed:',seed)
    set_seed(seed)

    dataset_train,dataset_val,dataset_test = data_loader_COVID19.load_finetune(args)

    if args.distributed:
        print('Using distributed mode')
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, seed=args.seed, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, seed=args.seed, shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, seed=args.seed, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and not args.test:
        logger.add(os.path.join(args.output_dir, args.save_dir,"loguru_log.txt"))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )
    
    if 'vit_base' in os.path.basename(args.finetune) or 'base' in args.finetune:
        args.model = 'vit_base_patch16'
    elif 'vit_large' in os.path.basename(args.finetune) or 'large' in args.finetune:
        args.model = 'vit_large_patch16'
    elif 'vit_huge' in os.path.basename(args.finetune) or 'huge' in args.finetune:
        args.model = 'vit_huge_patch14'

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    
    if args.finetune and not args.test:
        # pretrain_model_path = './ckpts_dir/medical_pretrained_models/MAE'
        checkpoint = torch.load(args.finetune, map_location='cpu')
        logger.info("Load pretrained checkpoint from: %s" % args.finetune)
        if 'CXC_resumed_pretrain' in args.finetune or 'CXC_continual_' in args.finetune or 'data13_mae_pretrain' in args.finetune or 'data14_mae_pretrain' in args.finetune \
            or 'data21_mae_pretrain' in args.finetune or 'data35_mae_pretrain' in args.finetune or 'data36_mae_pretrain' in args.finetune or 'allCT_mae_pretrain' in args.finetune \
            or 'deeplesion_mae_pretrain' in args.finetune or 'chexpert_mae_pretrain' in args.finetune or '7xray_mae_pretrain' in args.finetune \
            or ('checkpoint-' in args.finetune and 'checkpoint-best' not in args.finetune) or 'CXC_mae_pretrain_vit_' in args.finetune:
            logger.info('This is our own medical pretrained model.'.upper())
            checkpoint_model = {k:v for k, v in checkpoint['model'].items() if 'decoder_' not in k and 'mask_token' not in k}  #自己pretrain的model里面包含decoder部分和mask_token，需要删掉,
                                                                                                    #且包含norm.weight/norm.bias，经过finetune会被换成fc_norm.weight/fc_norm.bias,并加上head.weight/head.bias
            state_dict = model.state_dict()

            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

        elif os.path.basename(args.finetune) in ['mae_pretrain_vit_base.pth','mae_pretrain_vit_large.pth','mae_pretrain_vit_huge.pth']:
            print('This is MAE official pretrained model.'.upper())
            checkpoint_model = checkpoint['model']       
            state_dict = model.state_dict()

            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        else:
            print('This is (the intermediate or MAE official) finetuned model or TFS model.'.upper())
            checkpoint_model = checkpoint['model']
            for k in ['head.weight', 'head.bias','fc_norm.weight', 'fc_norm.bias']:
                if k in checkpoint_model:
                    print(f"Removing key {k} from (our own intermediate or MAE official) finetuned checkpoint.")
                    del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        
        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # block-wise freezing
        if (not args.attn and not args.mlp and not args.norm1 and not args.norm2 and not args.bias) and args.frozen_blocks > 0 and args.reinit_blocks == 0:
            print('BLOCK-WISE FREEZING.')
            freeze_blocks(args.frozen_blocks,model)

        # Full fine-tuning and block-wise re-initializing
        elif (not args.attn and not args.mlp and not args.norm1 and not args.norm2 and not args.bias) and args.frozen_blocks == 0 and args.reinit_blocks > 0:
            print('FULL FINE-TUNING AND RE-INIT BLOCKS.')
            reinit_blocks(args.reinit_blocks,model)
         
        # Partial fine-tuning
        elif args.attn or args.mlp or args.norm1 or args.norm2 or args.bias:
            print('PARTIAL part-wise FINE-TUNING.')
            partial_ft(args.attn,args.mlp,args.norm1,args.norm2,args.bias,model)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    if hasattr(model_without_ddp, 'no_weight_decay'):
        print('no_weight_decay_list:',model_without_ddp.no_weight_decay())
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(), # no_weight_decay_list = {'dist_token', 'cls_token', 'pos_embed'}; DeiT中distilled = True时才有dist_token;
        layer_decay=args.layer_decay                              # self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
    )

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.test:
        print('#'*30)
        test_stats = evaluate('test',data_loader_test, model, device)
        print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc']:.4f}")
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        print(log_stats)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    model_best = model

    logger.info('random_seed:{}; finetune dataset:{}; finetune model:{}'.format(args.seed, args.tar,args.finetune))

    stop = 0
    for epoch in range(args.start_epoch, args.epochs):
        stop += 1
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        print('-' * 30)
        print(f"Number of train images: {len(dataset_train)}")
        print('-' * 30)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            args=args
        )
        # Save a checkpoint every 'ckpt_save_freq' epochs
        if args.output_dir and args.save_all and epoch % args.ckpt_save_freq == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        print('-' * 30)
        print(f"Number of val images: {len(dataset_val)}")
        print('-' * 30)
        val_stats = evaluate('val',data_loader_val, model, device)

        # wandb.log({"train_loss": train_stats['loss'],"epoch": epoch})
        # wandb.log({"val_loss": val_stats['loss'],"epoch": epoch})
        # wandb.log({"val_acc": val_stats['acc'],"epoch": epoch})
        # wandb.log({"val_auc": val_stats['auc'],"epoch": epoch})
        # wandb.log({"val_f1": val_stats['f1'],"epoch": epoch})
        # wandb.log({"val_pre": val_stats['pre'],"epoch": epoch})
        # wandb.log({"val_recall": val_stats['recall'],"epoch": epoch})
        # wandb.log({"val_specificity": val_stats['specificity'],"epoch": epoch})
        # wandb.log({"val_CM": val_stats['CM(tn,fp,fn,tp)'],"epoch": epoch})

        train_log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                           **{f'val_{k}': v for k, v in val_stats.items()},
                           'epoch': epoch,
                           'n_parameters': n_parameters
                        }

        if args.output_dir and misc.is_main_process():
            logger.info(train_log_stats)

        # To determine when to stop early
        if val_stats["acc"] >= max_accuracy:
            stop = 0
            max_accuracy = val_stats["acc"]
            # Only save the best model
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch='best')
            model_best = model
            best_epoch = epoch

            print('-' * 30)
            print('best_epoch:', best_epoch)
            print(f'max_accuracy: {max_accuracy:.4f}')
            print('-' * 30,'\n')

        if stop > args.early_stop:
            break
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training completed in {} \n'.format(total_time_str))

    # Finish training, use the best model to do testing
    print('-' * 30)
    print(f"Number of test images: {len(dataset_test)}")
    print('-' * 30)
    test_stats = evaluate('test', data_loader_test, model_best, device)

    test_log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                    'best_epoch': best_epoch,
                    'best_val_acc':  max_accuracy,
                    'training_time': total_time_str
                    }
    logger.info(test_log_stats)

    # wandb.run.summary['test_loss'] = test_stats["loss"]
    # wandb.run.summary['test_acc'] = test_stats["acc"]
    # wandb.run.summary['test_auc'] = test_stats["auc"]
    # wandb.run.summary['test_f1'] = test_stats["f1"]
    # wandb.run.summary['test_CM(tn,fp,fn,tp)'] = test_stats["CM(tn,fp,fn,tp)"]
    # wandb.run.summary['best_epoch'] = best_epoch
    # wandb.run.summary['best_val_acc'] = max_accuracy
    # wandb.run.summary['n_parameters'] = n_parameters / 1.e6
    # wandb.run.summary['training_time'] = total_time_str
    
    # os.remove(os.path.join(args.output_dir, args.save_dir,'checkpoint-best.pth')) # not to save every fine-tuning checkpoint for wandb sweep runs in order to save device storage space

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # os.environ["WANDB_DIR"] = os.path.abspath(args.output_dir)
    # run = wandb.init(config = args, project="MAE_COVID19", entity="xxx",dir=args.output_dir,settings=wandb.Settings(start_method="fork"))

    if not args.finetune:
        print('Train from scratch with ViT.')
        args.tag = 'TFS'
        # run.tags = run.tags + ('TFS',)
    else:
        if (not args.attn and not args.mlp and not args.norm1 and not args.norm2 and not args.bias) and args.frozen_blocks == 0:
            print('PERFORM FULL FINE-TUNING with/without re-initializing.')
            args.tag = 'FFT'
            # run.tags = run.tags + ('FFT',)
        elif (not args.attn and not args.mlp and not args.norm1 and not args.norm2 and not args.bias) and args.frozen_blocks > 0 and args.reinit_blocks == 0:
            print('PERFORM block-wise FREEZING.')
            args.tag = 'PFR'
            # run.tags = run.tags + ('PFR',)
        else:
            print('PERFORM PARTIAL FINE-TUNING.')
            args.tag = 'PFT'
            # run.tags = run.tags + ('PFT',)

    if args.output_dir and not args.test:
        if args.tag == 'TFS':
            args.save_dir = os.path.join('output_tfs','TFS_with_' + args.model, args.split_ratio.strip(), args.tar + '_seed' + str(args.seed) \
                + '_bs' +str(args.batch_size) + '_b' +str(round(args.blr,5)) + '_l' +str(round(args.layer_decay,4)) + '_w' +str(round(args.weight_decay,4)) \
                + '_d' +str(round(args.drop_path,4)) + '_r' +str(round(args.reprob,4)) + '_m' +str(round(args.mixup,4)) + '_c' +str(round(args.cutmix,4)) + '_' + args.tag)
            Path(args.output_dir,args.save_dir).mkdir(parents=True, exist_ok=True)# parents：如果父目录不存在，是否创建父目录；exist:只有在目录不存在时创建目录，目录已存在时不会抛出异常。
        else:
            args.save_dir = os.path.join('output_finetune','finetune_with_'+os.path.basename(args.finetune).strip('.pth'), args.split_ratio.strip(), args.tar + '_seed' + str(args.seed) \
                + '_bs' +str(args.batch_size) + '_b' +str(round(args.blr,5)) + '_l' +str(round(args.layer_decay,4)) + '_w' +str(round(args.weight_decay,4)) \
                + '_d' +str(round(args.drop_path,4)) + '_r' +str(round(args.reprob,4)) + '_m' +str(round(args.mixup,4)) + '_c' +str(round(args.cutmix,4)) \
                + '_rb' +str(args.reinit_blocks) + '_fb' +str(args.frozen_blocks) + '_attn' + str(args.attn) + '_mlp' + str(args.mlp) + '_bias' + str(args.bias) + '_' + args.tag)
            Path(args.output_dir,args.save_dir).mkdir(parents=True, exist_ok=True)# parents：如果父目录不存在，是否创建父目录；exist:只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    else:
        args.save_dir = os.path.join('output_test','test_with_'+os.path.basename(args.finetune).strip('.pth'), args.split_ratio.strip(), args.tar + '_seed' + str(args.seed) \
                + '_bs' +str(args.batch_size) + '_b' +str(round(args.blr,5)) + '_l' +str(round(args.layer_decay,4)) + '_w' +str(round(args.weight_decay,4)) \
                + '_d' +str(round(args.drop_path,4)) + '_r' +str(round(args.reprob,4)) + '_m' +str(round(args.mixup,4)) + '_c' +str(round(args.cutmix,4)) \
                + '_rb' +str(args.reinit_blocks) + '_fb' +str(args.frozen_blocks) + '_attn' + str(args.attn) + '_mlp' + str(args.mlp) + '_bias' + str(args.bias) + '_' + args.tag)
        Path(args.output_dir,args.save_dir).mkdir(parents=True, exist_ok=True)# parents：如果父目录不存在，是否创建父目录；exist:只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    main(args)