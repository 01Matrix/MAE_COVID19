# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

from cgi import test
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

import numpy as np

from timm.data import Mixup
from timm.utils import accuracy

from sklearn.preprocessing import label_binarize

import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, roc_auc_score, f1_score
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(phase, data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    if phase == 'test':
        header = 'Test:'
    elif phase == 'val':
        header = 'val'

    # switch to evaluation mode
    model.eval()

    target_list = []
    output_list = []
    score_list = []
    best_acc = 0

    for batch in metric_logger.log_every(data_loader, 1, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        softmax_score = F.softmax(output, dim = 1)
        output = output.argmax(dim=1, keepdim=True)
        target = target.unsqueeze(-1)

        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        target_list = np.append(target_list, target)
        output_list = np.append(output_list, output)
        
        softmax_score = softmax_score.cpu().detach().numpy()
        score_list = np.append(score_list, softmax_score[:, 1])

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    tn, fp, fn, tp = confusion_matrix(target_list,output_list).ravel()
    acc = accuracy_score(target_list, output_list)
    Precision = precision_score(target_list, output_list)# precision = tp/(tp+fp)
    Recall = recall_score(target_list, output_list) # recall = tp/(tp+fn)=sensitivity
    f1 = f1_score(target_list, output_list)
    AUC_score = roc_auc_score(target_list, score_list)
    Specificity = tn/(tn+fp)
    # import pdb;pdb.set_trace()

    stats_dict={
    'acc':acc,
    'pre':Precision,# precision = tp/(tp+fp)
    'recall':Recall,# recall = tp/(tp+fn)=sensitivity
    'f1':f1,
    'auc':AUC_score,
    'specificity':Specificity,
    'CM(tn,fp,fn,tp)':'[{} {} {} {}]'.format(tn, fp, fn, tp),
    'loss':metric_logger.loss.global_avg
    }

    return stats_dict