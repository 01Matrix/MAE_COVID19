# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import wandb
import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop
import util.data_loader_COVID19 as data_loader_COVID19

import models_vit

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', type=str2bool, default=False,
                        help='Use class token instead of global pool for classification')
    # parser.add_argument('--cls_token', action='store_false', dest='global_pool',
    #                     help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--split_ratio',default='2:3:5',type=str,help=' Split dataset to train:val:test')
    parser.add_argument('--tar', type=str, default='U_orig', help='finetune data')

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='/sharefs/baaihealth/xiaohongwang/MAE_COVID19/output_linprobe',
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
    # parser.add_argument('--eval', action='store_true',
                        # help='Perform evaluation only')
    parser.add_argument('--test', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
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

def main(args):
    
    
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    set_seed(args.seed)
    # # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # cudnn.benchmark = True

    dataset_train,dataset_val,dataset_test = data_loader_COVID19.load_linprobe(args)
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    # print(dataset_train)
    # print(dataset_val)

    if False:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # if global_rank == 0 and args.log_dir is not None and not args.eval:
    if not args.test:
        # os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.output_dir,args.save_dir))
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn
    )

    if 'vit_large' in os.path.basename(args.finetune) or 'large' in args.finetune:
        args.model = 'vit_large_patch16'

    if 'vit_huge' in os.path.basename(args.finetune) or 'huge' in args.finetune:
        args.model = 'vit_huge_patch14'

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.test:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if 'C_model_resumed_pretrain' in args.finetune or 'MAE_checkpoint_799' in args.finetune or 'pretrain/checkpoint-' in args.finetune:
            print('This is our own pretrained model.')
            checkpoint_model = {k:v for k, v in checkpoint['model'].items() if 'decoder_' not in k and 'mask_token' not in k}  #自己pretrain的model里面包含decoder部分和mask_token，需要删掉,
                                                                                                        #且包含norm.weight/norm.bias，经过finetune(global_pool=True)会被换成fc_norm.weight/fc_norm.bias,并加上head.weight/head.bias
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
            if os.path.basename(args.finetune) in ['C_orig_8:1:1_mae_pretrain_vit_base.pth','C_orig_8:1:1_mae_pretrain_vit_large.pth']:
                print('@@@@@@@@@@@')
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'norm.weight', 'norm.bias'}
            else:
                print('!!!!!!!!!!!')
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)

    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

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

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # if args.eval:
    #     val_stats = evaluate('val',data_loader_val, model, device)
    #     print(f"Accuracy of the network on the {len(dataset_val)} val images: {val_stats['acc']:.4f}")
    #     exit(0)

    if args.test:
        print('#'*20)
        test_stats = evaluate('test',data_loader_test, model, device)
        print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc']:.4f}")
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        print(log_stats)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    model_best = model

    with open(os.path.join(args.output_dir, args.save_dir,"lin_probe_train_log.txt"), mode="a", encoding="utf-8") as f:
        f.write("\n" + 'lin_probe--' + 'finetune dataset:{}; finetune model:{}'.format(args.tar,args.finetune) + "\n")
    stop  =0
    for epoch in range(args.start_epoch, args.epochs):
        stop += 1
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and epoch % 50 == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        val_stats = evaluate('val',data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} val images: {val_stats['acc']:.4f}")
        max_accuracy = max(max_accuracy, val_stats["acc"])
        print(f'Max accuracy: {max_accuracy:.4f}')
        if val_stats["acc"] >= max_accuracy:
            stop = 0
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
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir,args.save_dir, "lin_probe_train_log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(train_log_stats) + "\n")
        if stop > args.early_stop:
            break
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    print(f"Number of test images: {len(dataset_test)}")
    test_stats = evaluate('test', data_loader_test, model_best, device)
    test_log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                    'Total epoch': epoch+1,
                    'Best epoch': best_epoch+1,
                    'Best val_acc':  max_accuracy,
                    'Training time':total_time_str
                    }
    print(test_log_stats)

    wandb.run.summary['test_acc'] = test_stats["acc"]
    wandb.run.summary['test_loss'] = test_stats["loss"]
    wandb.run.summary['test_f1'] = test_stats["f1"]
    wandb.run.summary['test_auc'] = test_stats["auc"]
    wandb.run.summary['CM(tn,fp,fn,tp)'] = test_stats["CM(tn,fp,fn,tp)"]
    wandb.run.summary['Total_epoch'] = epoch

    with open(os.path.join(args.output_dir, args.save_dir,"lin_probe_test_log.txt"), mode="a", encoding="utf-8") as f:
        f.write("\n" + 'lin_probe--'+ 'finetune dataset:{}; finetune model:{}'.format(args.tar,args.finetune) + "\n")
        f.write(json.dumps(test_log_stats) + "\n")
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.global_pool = args.cls_token # False

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
        print('Perform linear probing.')
        args.tag = 'LP'
        run.tags = run.tags + ('LP',)
     

    if args.output_dir and not args.test:
        if args.tag == 'TFS':
            args.save_dir = os.path.join(args.split_ratio.strip(),'TFS_with_' + args.model, args.tar + '_seed' + str(args.seed) \
                + '_bs' +str(args.batch_size) + '_b' +str(round(args.blr,5)) + '_l' +str(round(args.layer_decay,4)) + '_w' +str(round(args.weight_decay,4)) \
                + '_d' +str(round(args.drop_path,4)) + '_r' +str(round(args.reprob,4)) + '_m' +str(round(args.mixup,4)) + '_c' +str(round(args.cutmix,4)) + '_bl' \
                + '_'.join(args.block_list.split(',')) + '_fft' + str(args.fft) + '_attn' + str(args.attn) + '_mlp' + str(args.mlp) + '_' + args.tag)
        else:
            args.save_dir = os.path.join(args.split_ratio.strip(),'linprobe_with_'+os.path.basename(args.finetune).strip('.pth'), args.tar + '_seed' + str(args.seed) \
                + '_bs' +str(args.batch_size) + '_b' +str(round(args.blr,5)) + '_' + args.tag)
        Path(args.output_dir,args.save_dir).mkdir(parents=True, exist_ok=True)# parents：如果父目录不存在，是否创建父目录；exist:只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    main(args)