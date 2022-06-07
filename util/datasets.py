# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
from datasets.covid_ct_dataset import COVID_CT_Dataset
from datasets.covid_ct_dataset_txt import COVID_CT_Dataset_txt

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import InterpolationMode

def build_dataset(is_train, args, test=False):
    transform = build_transform(is_train, args)

    if args.test or test:
        root = os.path.join(args.data_path, 'test')
    else:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')

    dataset = COVID_CT_Dataset(root, transform=transform)

    print(dataset)

    return dataset

def build_dataset_txt(is_train, args, data_path):
    transform = build_transform(is_train, args)

    dataset = COVID_CT_Dataset_txt(data_path, transform=transform)
    print(dataset)

    return dataset

def build_transform(is_train, args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
