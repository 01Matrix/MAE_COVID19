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
import datetime
from distutils.command.config import config
import json
# from random import shuffle
import numpy as np
import random
import wandb
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
# from datasets.covid_ct_dataset_txt import COVID_CT_Dataset_txt
import util.data_loader_COVID19 as data_loader_COVID19
import timm
assert timm.__version__ == "0.5.4" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
# from util.datasets import build_dataset, build_dataset_txt
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit
from engine_finetune_prompt import train_one_epoch, evaluate
# import fastfood
import prompters


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--early_stop', type=int, default=20)
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
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--split_ratio',default='2:3:5',type=str,help='Split dataset to train:val:test')
    parser.add_argument('--tar', type=str, default='U_sani2', help='finetune data')

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='/sharefs/baaihealth/xiaohongwang/MAE_COVID19/output_finetune',
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
    parser.add_argument('--block_list',type=str,default='', help='re-init weight of blocks list')
    parser.add_argument('--fft',type=str2bool, default=False,help='It means Full-finetuning')
    parser.add_argument('--attn',type=str2bool, default=False,help='It means just finetune attention layer.')
    parser.add_argument('--mlp',type=str2bool, default=False,help='It means just finetune mlp layer')
    parser.add_argument('--bias',type=str2bool, default=False,help='It means just finetune mlp layer')
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
    # torch.use_deterministic_algorithms(True) #这句话相当于自检，只要使用了不可复现的运算操作，代码就会自动报错。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

GLOBAL_SEED =1 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

# Layer-wise finetuning.
def layerwise_ft(flag_attn,flag_mlp,flag_bias,model):
    if not flag_attn and not flag_mlp and flag_bias:
        print('Just finetune bias layer and head.')
        for n, p in model.named_parameters():
            p.requires_grad = False
            if 'bias' in n:
                p.requires_grad = True
            if 'head' in n:
                p.requires_grad = True
            if 'fc_norm' in n:
                p.requires_grad = True
        
    if flag_attn and not flag_mlp:
        print('Just finetune attention layer and head.')
        for n, p in model.named_parameters():
            p.requires_grad = False
            if 'attn' in n:
                p.requires_grad = True
            if 'head' in n:
                p.requires_grad = True
            if 'fc_norm' in n:
                p.requires_grad = True
    if not flag_attn and flag_mlp:
        print('Just finetune MLP layer and head.')
        for n, p in model.named_parameters():
            p.requires_grad = False
            if 'mlp' in n:
                p.requires_grad = True
            if 'head' in n:
                p.requires_grad = True
            if 'fc_norm' in n:
                p.requires_grad = True
    if flag_attn and flag_mlp:
        print('Finetune attention and MLP layer, and head.')
        for n, p in model.named_parameters():
            p.requires_grad = False
            if 'attn' in n:
                p.requires_grad = True
            if 'mlp' in n:
                p.requires_grad = True
            if 'head' in n:
                p.requires_grad = True
            if 'fc_norm' in n:
                p.requires_grad = True
    if not flag_attn and not flag_mlp and not flag_bias: #接近于linear probe,但是和LP不同的是fc_norm层是unfreeze的
        print('Not Finetune attention and MLP layer, just finetune head, and freeze other layers.')
        for n, p in model.named_parameters():
            p.requires_grad = False
            if 'head' in n:
                p.requires_grad = True
            if 'fc_norm' in n:
                p.requires_grad = True
    re_init_w(flag_attn,flag_mlp,flag_bias,model)

# Weight re_init 权重重新初始化
def re_init_w(flag_attn,flag_mlp,flag_bias,model):
    if flag_bias:
        # trunc_normal_(model.patch_embed.proj.bias,std=2e-5)
        trunc_normal_(model.fc_norm.bias,std=2e-5)
        trunc_normal_(model.head.bias,std=2e-5)
    if args.block_list != '':
        for i in list(map(lambda x: int(x),args.block_list.split(','))): #对几个block的attention和mlp的权重进行重新初始化
            if flag_attn:
                print(f'Re-init weight of blocks[{i}].attn')
                trunc_normal_(model.blocks[i].attn.qkv.weight,std=2e-5)
                # trunc_normal_(model_without_ddp.blocks[i].attn.proj.weight,std=2e-5)
            if flag_mlp:
                print(f'Re-init weight of blocks[{i}].mlp')
                trunc_normal_(model.blocks[i].mlp.fc1.weight,std=2e-5)
                # trunc_normal_(model_without_ddp.blocks[i].mlp.fc2.weight,std=2e-5)
            if flag_bias:
                print(f'Re-init bias of patch_embed.proj and blocks[{i}]')
                trunc_normal_(model.blocks[i].norm1.bias,std=2e-5)
                trunc_normal_(model.blocks[i].attn.qkv.bias,std=2e-5)
                trunc_normal_(model.blocks[i].attn.proj.bias,std=2e-5)
                trunc_normal_(model.blocks[i].norm2.bias,std=2e-5)
                trunc_normal_(model.blocks[i].mlp.fc1.bias,std=2e-5)
                trunc_normal_(model.blocks[i].mlp.fc2.bias,std=2e-5)

            # if not flag_attn and not flag_mlp:
            #     print('Not Re-init weight of blocks.')
    else:
        print('Not Re-init weight of blocks.')

def main(args):
    
    # misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    set_seed(args.seed)
    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # cudnn.benchmark = True

    dataset_train,dataset_val,dataset_test = data_loader_COVID19.load_finetune(args)
    # import pdb;pdb.set_trace()

    if False:  # args.distributed:
        print('distributed')
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
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # if global_rank == 0 and not args.test:
    if not args.test:
        # os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.output_dir,args.save_dir))
    else:
        log_writer = None

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
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    if 'vit_large' in os.path.basename(args.finetune) or 'large' in args.finetune:
        args.model = 'vit_large_patch16'

    if 'vit_huge' in os.path.basename(args.finetune) or 'huge' in args.finetune:
        args.model = 'vit_huge_patch14'

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.test:
        # pretrain_model_path = '/sharefs/baaihealth/xiaohongwang/medical_pretrained_models/MAE/'
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pretrained checkpoint from: %s" % args.finetune)
        if 'C_model_resumed_pretrain' in args.finetune or 'MAE_checkpoint_799' in args.finetune or 'pretrain/checkpoint-' in args.finetune:
            print('This is our own pretrained model.')
            checkpoint_model = {k:v for k, v in checkpoint['model'].items() if 'decoder_' not in k and 'mask_token' not in k}  #自己pretrain的model里面包含decoder部分和mask_token，需要删掉,
                                                                                                    #且包含norm.weight/norm.bias，经过finetune会被换成fc_norm.weight/fc_norm.bias,并加上head.weight/head.bias
            state_dict = model.state_dict()

            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

        elif os.path.basename(args.finetune) in ['mae_pretrain_vit_base.pth','mae_pretrain_vit_large.pth','mae_pretrain_vit_huge.pth']:
            print('This is MAE official pretrained model.')
            checkpoint_model = checkpoint['model']       
            state_dict = model.state_dict()

            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        else:
            print('This is (our own intermediate or MAE official) finetuned model.')
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

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    # visual prompt
    prompter = prompters.__dict__[args.method](args).to(device)
    prompter.load_state_dict(checkpoint_model, strict=False)
    

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    # 不是full-finetuning, 而是只finetune MLP层 or attention层
    if args.finetune and not args.fft:
        print('Layer-wise finetuning.')
        layerwise_ft(args.attn,args.mlp,args.bias,model_without_ddp)

    # build optimizer with layer-wise lr decay (lrd)
    # if hasattr(model_without_ddp, 'no_weight_decay'):
    #     print('no_weight_decay_list',model_without_ddp.no_weight_decay())
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(), # no_weight_decay_list = {'dist_token', 'cls_token', 'pos_embed'}; DeiT中distilled = True时才有dist_token;
        layer_decay=args.layer_decay                              # self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
    )
    # model_without_ddp = fastfood.FastfoodWrap(model_without_ddp,intrinsic_dimension=5000, device=0)

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

    # if args.eval:
    #     print('-'*20)
    #     val_stats = evaluate('val',data_loader_val, model, device)
    #     print(f"Accuracy of the network on the {len(dataset_val)} val images: {val_stats['acc']:.4f}")
    #     exit(0)
   
    if args.test:
        print('#'*20)
        test_stats = evaluate('test',data_loader_test, model,prompter, device)
        print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc']:.4f}")
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        print(log_stats)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    model_best = model

    with open(os.path.join(args.output_dir, args.save_dir,"log.txt"), mode="w", encoding="utf-8") as f:
        f.write('random_seed:{}; finetune dataset:{}; finetune model:{}'.format(args.seed, args.tar,args.finetune) + "\n")

    stop = 0
    for epoch in range(args.start_epoch, args.epochs):
        stop += 1
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, prompter, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and args.save_all and epoch % 50 == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        val_stats = evaluate('val',data_loader_val, model,prompter,device)

        print(f"Number of val images: {len(dataset_val)}")
        # max_accuracy = max(max_accuracy, test_stats["acc1"])
        if val_stats["acc"] >= max_accuracy:
            stop = 0
            wandb.run.summary['best_val_acc'] = val_stats["acc"]
            max_accuracy = val_stats["acc"]
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch='best')
            model_best = model
            best_epoch = epoch

            print('-' * 20)
            print('Best epoch:', best_epoch)
            print(f'Max accuracy: {max_accuracy:.4f}')
            print('-' * 20,'\n')

        if log_writer is not None:
            log_writer.add_scalar('perf/val_acc', val_stats['acc'], epoch)
            log_writer.add_scalar('perf/val_pre', val_stats['pre'], epoch)
            log_writer.add_scalar('perf/val_recall', val_stats['recall'], epoch)
            log_writer.add_scalar('perf/val_f1', val_stats['f1'], epoch)
            log_writer.add_scalar('perf/val_auc', val_stats['auc'], epoch)
            log_writer.add_scalar('perf/val_specificity', val_stats['specificity'], epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)

        wandb.log({"train_loss": train_stats['loss'],"epoch": epoch})
        wandb.log({"val_loss": val_stats['loss'],"epoch": epoch})
        wandb.log({"val_acc": val_stats['acc'],"epoch": epoch})

        train_log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters
                            }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, args.save_dir,"log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(train_log_stats) + "\n")

        if stop > args.early_stop:
            break
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training completed in {} \n'.format(total_time_str))

    print(f"Number of test images: {len(dataset_test)}")
    test_stats = evaluate('test', data_loader_test, model_best,prompter, device)
    test_log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                    'Total epoch': epoch,
                    'Best epoch': best_epoch,
                    'Best val_acc':  max_accuracy,
                    'Training time': total_time_str
                    }
    print(test_log_stats)

    wandb.run.summary['test_acc'] = test_stats["acc"]
    wandb.run.summary['test_loss'] = test_stats["loss"]
    wandb.run.summary['test_f1'] = test_stats["f1"]
    wandb.run.summary['test_auc'] = test_stats["auc"]
    wandb.run.summary['CM(tn,fp,fn,tp)'] = test_stats["CM(tn,fp,fn,tp)"]
    wandb.run.summary['Total_epoch'] = epoch
    # wandb.run.summary['Training_time'] = total_time_str

    with open(os.path.join(args.output_dir, args.save_dir,"log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(test_log_stats) + "\n")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # os.environ["WANDB_DIR"] = os.path.abspath("/sharefs/baaihealth/xiaohongwang/MAE_COVID19")
    run = wandb.init(config = args, project="MAE_COVID19_2", entity="bluedynamic",dir='/sharefs/baaihealth/xiaohongwang/MAE_COVID19')
    # api = wandb.Api()
    # run_id = run.id
    # run = api.run("bluedynamic/MAE_COVID19/{}".format(run_id))

    if not args.finetune:
        print('Train from scratch with ViT.')
        args.tag = 'TFS'
        run.tags = run.tags + ('TFS',)
    else:
        if args.fft:
            print('Perform full finetuning, update all the model parameters.')
            args.tag = 'FFT'
            run.tags = run.tags + ('FFT',)
            # run.config['finetune'] = os.path.basename(args.finetune)
            # run.update()

        else:
            print('Perform layer-wise finetuning.')
            args.tag = 'LFT'
            run.tags = run.tags + ('LFT',)
            # run.config['finetune'] = os.path.basename(args.finetune)
            # run.update()

    if args.output_dir and not args.test:
        if args.tag == 'TFS':
            args.save_dir = os.path.join(args.split_ratio.strip(),'TFS_with_' + args.model, args.tar + '_seed' + str(args.seed) \
                + '_bs' +str(args.batch_size) + '_b' +str(round(args.blr,5)) + '_l' +str(round(args.layer_decay,4)) + '_w' +str(round(args.weight_decay,4)) \
                + '_d' +str(round(args.drop_path,4)) + '_r' +str(round(args.reprob,4)) + '_m' +str(round(args.mixup,4)) + '_c' +str(round(args.cutmix,4)) + '_bl' \
                + '_'.join(args.block_list.split(',')) + '_fft' + str(args.fft) + '_attn' + str(args.attn) + '_mlp' + str(args.mlp) + '_' + args.tag)
        else:
            args.save_dir = os.path.join(args.split_ratio.strip(),'finetune_with_'+os.path.basename(args.finetune).strip('.pth'), args.tar + '_seed' + str(args.seed) \
                + '_bs' +str(args.batch_size) + '_b' +str(round(args.blr,5)) + '_l' +str(round(args.layer_decay,4)) + '_w' +str(round(args.weight_decay,4)) \
                + '_d' +str(round(args.drop_path,4)) + '_r' +str(round(args.reprob,4)) + '_m' +str(round(args.mixup,4)) + '_c' +str(round(args.cutmix,4)) + '_bl' \
                + '_'.join(args.block_list.split(',')) + '_fft' + str(args.fft) + '_attn' + str(args.attn) + '_mlp' + str(args.mlp) + '_' + args.tag)
        Path(args.output_dir,args.save_dir).mkdir(parents=True, exist_ok=True)# parents：如果父目录不存在，是否创建父目录；exist:只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    main(args)

    # Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:
    # 1. Define the sweep: we do this by creating a dictionary or a YAML file that specifies the parameters to search through, the search strategy, the optimization metric et all.
    # 2. Initialize the sweep: with one line of code we initialize the sweep and pass in the dictionary of sweep configurations: sweep_id = wandb.sweep(sweep_config)
    # 3. Run the sweep agent: also accomplished with one line of code, we call wandb.agent() and pass the sweep_id to run, 
    # along with a function that defines your model architecture and trains it: wandb.agent(sweep_id, function=train)
    
    # 超参数搜索方法，可以选择：grid random bayes
    # sweep_config = {
    #     'method': 'random'
    #     }
    # # metric = {
    # # 'name': 'loss',
    # # 'goal': 'minimize'   
    # # }
    # # sweep_config['metric'] = metric
    # # 参数范围
    # parameters_dict = {
    #     'blr': {
    #         # a flat distribution between 0 and 0.1
    #         'distribution': 'uniform',
    #         'min': 0,
    #         'max': 0.01
    #     },
    #     'layer_decay': {
    #         # a flat distribution between 0 and 0.1
    #         'distribution': 'uniform',
    #         'min': 0.5,
    #         'max': 0.9
    #     },
    #     'weight_decay': {
    #         # a flat distribution between 0 and 0.1
    #         'distribution': 'uniform',
    #         'min': 0,
    #         'max': 0.1
    #     },
    #     'drop_path': {
    #         # a flat distribution between 0 and 0.1
    #         'distribution': 'uniform',
    #         'min': 0,
    #         'max': 0.1
    #     },
    #     'reprob': {
    #         # a flat distribution between 0 and 0.1
    #         'distribution': 'uniform',
    #         'min': 0,
    #         'max': 0.1
    #     },
    #     'mixup': {
    #         # a flat distribution between 0 and 0.1
    #         'distribution': 'uniform',
    #         'min': 0,
    #         'max': 0.1
    #     },
    #     'cutmix': {
    #         # a flat distribution between 0 and 0.1
    #         'distribution': 'uniform',
    #         'min': 0,
    #         'max': 0.1
    #     },
    #     'batch_size': {
    #         # integers between 4 and 32
    #         # with evenly-distributed logarithms 
    #         'distribution': 'q_log_uniform',
    #         'q': 1,
    #         'min': 4,
    #         'max': 32,
    #     }
    # }

    # sweep_config['parameters'] = parameters_dict
    # sweep_id = wandb.sweep(sweep_config, project="MAE_COVID19")
    # # main(args)
    # wandb.agent(sweep_id, main(args), count=3)

    # 在以下几种情况中,head和fc_norm层均不冻结
    #1. args.fft==True: full fientuning（FFT）: 解冻所有层,除head外,其他层的权重不重新初始化（re-init weight）
    #2. args.fft == False: 称之为layer-wise finetuning（LFT）,包含以下几种情况
        #2.1 args.attn==True and args.mlp==False: 解冻attention层，head层和fc_norm层；除head外,对attn.qkv层选择是否re-init weight以及re-init哪一个block（with or w/o weight re-init）,
        #2.2 args.attn==False and args.mlp==True: 解冻mlp层，head层和fc_norm层；除head外,对mlp.fc1层选择是否re-init weight以及re-init哪一个block（with or w/o weight re-init）,
        #2.3 args.attn==True and args.mlp==True: 解冻attention层，mlp层，head层和fc_norm层；除head外,对attn.qkv层和mlp.fc1层选择是否re-init weight以及re-init哪一个block（with or w/o weight re-init）,
        #2.4 args.attn==False and args.mlp==False: 仅解冻head层和fc_norm层；除head外,其他层的权重不重新初始化（re-init weight）
        # block_list是需要进行weight re-init的block的序号（vit_base是0-11,vit_large是0-23）
        # if args.block_list为空，上述的2.1/2.2/2.3中对应的re-init weight都不执行
    #3. args.bias=True, 只finetune bias层