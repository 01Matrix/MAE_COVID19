# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# # --------------------------------------------------------
# # References:
# # DeiT: https://github.com/facebookresearch/deit
# # BEiT: https://github.com/microsoft/unilm/tree/master/beit
# # --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import torch.distributed as dist
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    # dist.barrier()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        print('-' * 20)
        print('data_iter_step: ', data_iter_step)
        print('-' * 20)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# # --------------------------------------------------------
# # Based on BEiT, timm, DINO and DeiT code bases
# # https://github.com/microsoft/unilm/tree/master/beit
# # https://github.com/rwightman/pytorch-image-models/tree/master/timm
# # https://github.com/facebookresearch/deit
# # https://github.com/facebookresearch/dino
# # --------------------------------------------------------'
# import math
# import sys
# from typing import Iterable

# import torch
# import torch.nn as nn

# import util.misc as utils
# from einops import rearrange
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# from tensorboardX import SummaryWriter

# def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
#                     normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
#                     lr_schedule_values=None, wd_schedule_values=None, args=None):
#     model.train()
    
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10

#     loss_func = nn.MSELoss()


#     for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         # assign learning rate & weight decay for each step
#         it = start_steps + step  # global training iteration
#         if lr_schedule_values is not None or wd_schedule_values is not None:
#             for i, param_group in enumerate(optimizer.param_groups):
#                 if lr_schedule_values is not None:
#                     param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
#                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#                     param_group["weight_decay"] = wd_schedule_values[it]

#         images, bool_masked_pos = batch
#         images = images.to(device, non_blocking=True)
#         bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

#         # import pdb; pdb.set_trace()
#         with torch.no_grad():
#             # calculate the predict label
#             mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
#             std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
#             unnorm_images = images * std + mean  # in [0, 1]

#             if normlize_target:
#                 images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
#                 images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
#                     ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
#                 # we find that the mean is about 0.48 and standard deviation is about 0.08.
#                 images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
#             else:
#                 images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

#             B, _, C = images_patch.shape
#             labels = images_patch[bool_masked_pos].reshape(B, -1, C)

#         with torch.cuda.amp.autocast():
#             outputs = model(images, bool_masked_pos)
#             loss = loss_func(input=outputs, target=labels)

#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)

#         optimizer.zero_grad()
#         # this attribute is added by timm on one optimizer (adahessian)
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
#                                 parameters=model.parameters(), create_graph=is_second_order)
#         loss_scale_value = loss_scaler.state_dict()["scale"]

#         torch.cuda.synchronize()

#         metric_logger.update(loss=loss_value)
#         metric_logger.update(loss_scale=loss_scale_value)
#         min_lr = 10.
#         max_lr = 0.
#         for group in optimizer.param_groups:
#             min_lr = min(min_lr, group["lr"])
#             max_lr = max(max_lr, group["lr"])

#         metric_logger.update(lr=max_lr)
#         metric_logger.update(min_lr=min_lr)
#         weight_decay_value = None
#         for group in optimizer.param_groups:
#             if group["weight_decay"] > 0:
#                 weight_decay_value = group["weight_decay"]
#         metric_logger.update(weight_decay=weight_decay_value)
#         metric_logger.update(grad_norm=grad_norm)

#         if log_writer is not None:
#             log_writer.update(loss=loss_value, head="loss")
#             log_writer.update(loss_scale=loss_scale_value, head="opt")
#             log_writer.update(lr=max_lr, head="opt")
#             log_writer.update(min_lr=min_lr, head="opt")
#             log_writer.update(weight_decay=weight_decay_value, head="opt")
#             log_writer.update(grad_norm=grad_norm, head="opt")

#             log_writer.set_step()

#         if lr_scheduler is not None:
#             lr_scheduler.step_update(start_steps + step)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
