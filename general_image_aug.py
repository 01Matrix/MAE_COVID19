import random

import numpy as np

from PIL import Image, ImageOps, ImageFilter

from skimage.filters import gaussian

import torch

import math

import numbers

import random

class RandomVerticalFlip(object):

    def __call__(self, img):

        if random.random() < 0.5:

            return img.transpose(Image.FLIP_TOP_BOTTOM)

        return img

class DeNormalize(object):

    def __init__(self, mean, std):

        self.mean = mean

        self.std = std

    def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):

            t.mul_(s).add_(m)

        return tensor

class MaskToTensor(object):

    def __call__(self, img):

        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class FreeScale(object):

    def __init__(self, size, interpolation=Image.BILINEAR):

        self.size = tuple(reversed(size))  # size: (h, w)

        self.interpolation = interpolation

    def __call__(self, img):

        return img.resize(self.size, self.interpolation)

class FlipChannels(object):

    def __call__(self, img):

        img = np.array(img)[:, :, ::-1]

        return Image.fromarray(img.astype(np.uint8))

class RandomGaussianBlur(object):

    def __call__(self, img):

        sigma = 0.15 + random.random() * 1.15

        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)

        blurred_img *= 255

        return Image.fromarray(blurred_img.astype(np.uint8))

# 组合

class Compose(object):

    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, img, mask):

        assert img.size == mask.size

        for t in self.transforms:

            img, mask = t(img, mask)

        return img, mask

# 随机裁剪

class RandomCrop(object):

    def __init__(self, size, padding=0):

        if isinstance(size, numbers.Number):

            self.size = (int(size), int(size))

        else:

            self.size = size

        self.padding = padding

    def __call__(self, img, mask):

        if self.padding > 0:

            img = ImageOps.expand(img, border=self.padding, fill=0)

            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size

        w, h = img.size

        th, tw = self.size

        if w == tw and h == th:

            return img, mask

        if w < tw or h < th:

            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)

        y1 = random.randint(0, h - th)

        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

#  中心裁剪

class CenterCrop(object):

    def __init__(self, size):

        if isinstance(size, numbers.Number):

            self.size = (int(size), int(size))

        else:

            self.size = size

    def __call__(self, img, mask):

        assert img.size == mask.size

        w, h = img.size

        th, tw = self.size

        x1 = int(round((w - tw) / 2.))

        y1 = int(round((h - th) / 2.))

        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class RandomHorizontallyFlip(object):

    def __call__(self, img, mask):

        if random.random() < 0.5:

            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask

class Scale(object):

    def __init__(self, size):

        self.size = size

    def __call__(self, img, mask):

        assert img.size == mask.size

        w, h = img.size

        if (w >= h and w == self.size) or (h >= w and h == self.size):

            return img, mask

        if w > h:

            ow = self.size

            oh = int(self.size * h / w)

            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)

        else:

            oh = self.size

            ow = int(self.size * w / h)

            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)

class RandomSizedCrop(object):

    def __init__(self, size):

        self.size = size

    def __call__(self, img, mask):

        assert img.size == mask.size

        for attempt in range(10):

            area = img.size[0] * img.size[1]

            target_area = random.uniform(0.45, 1.0) * area

            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))

            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:

                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:

                x1 = random.randint(0, img.size[0] - w)

                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))

                mask = mask.crop((x1, y1, x1 + w, y1 + h))

                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),

                                                                                       Image.NEAREST)

        # Fallback

        scale = Scale(self.size)

        crop = CenterCrop(self.size)

        return crop(*scale(img, mask))

class RandomRotate(object):

    def __init__(self, degree):

        self.degree = degree

    def __call__(self, img, mask):

        rotate_degree = random.random() * 2 * self.degree - self.degree

        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

class RandomSized(object):

    def __init__(self, size):

        self.size = size

        self.scale = Scale(self.size)

        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):

        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])

        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))

class SlidingCropOld(object):

    def __init__(self, crop_size, stride_rate, ignore_label):

        self.crop_size = crop_size

        self.stride_rate = stride_rate

        self.ignore_label = ignore_label

    def _pad(self, img, mask):

        h, w = img.shape[: 2]

        pad_h = max(self.crop_size - h, 0)

        pad_w = max(self.crop_size - w, 0)

        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')

        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)

        return img, mask

    def __call__(self, img, mask):

        assert img.size == mask.size

        w, h = img.size

        long_size = max(h, w)

        img = np.array(img)

        mask = np.array(mask)

        if long_size > self.crop_size:

            stride = int(math.ceil(self.crop_size * self.stride_rate))

            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1

            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1

            img_sublist, mask_sublist = [], []

            for yy in range(h_step_num):

                for xx in range(w_step_num):

                    sy, sx = yy * stride, xx * stride

                    ey, ex = sy + self.crop_size, sx + self.crop_size

                    img_sub = img[sy: ey, sx: ex, :]

                    mask_sub = mask[sy: ey, sx: ex]

                    img_sub, mask_sub = self._pad(img_sub, mask_sub)

                    img_sublist.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))

                    mask_sublist.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))

            return img_sublist, mask_sublist

        else:

            img, mask = self._pad(img, mask)

            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')

            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')

            return img, mask

class SlidingCrop(object):

    def __init__(self, crop_size, stride_rate, ignore_label):

        self.crop_size = crop_size

        self.stride_rate = stride_rate

        self.ignore_label = ignore_label

    def _pad(self, img, mask):

        h, w = img.shape[: 2]

        pad_h = max(self.crop_size - h, 0)

        pad_w = max(self.crop_size - w, 0)

        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')

        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)

        return img, mask, h, w

    def __call__(self, img, mask):

        assert img.size == mask.size

        w, h = img.size

        long_size = max(h, w)

        img = np.array(img)

        mask = np.array(mask)

        if long_size > self.crop_size:

            stride = int(math.ceil(self.crop_size * self.stride_rate))

            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1

            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1

            img_slices, mask_slices, slices_info = [], [], []

            for yy in range(h_step_num):

                for xx in range(w_step_num):

                    sy, sx = yy * stride, xx * stride

                    ey, ex = sy + self.crop_size, sx + self.crop_size

                    img_sub = img[sy: ey, sx: ex, :]

                    mask_sub = mask[sy: ey, sx: ex]

                    img_sub, mask_sub, sub_h, sub_w = self._pad(img_sub, mask_sub)

                    img_slices.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))

                    mask_slices.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))

                    slices_info.append([sy, ey, sx, ex, sub_h, sub_w])

            return img_slices, mask_slices, slices_info

        else:

            img, mask, sub_h, sub_w = self._pad(img, mask)

            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')

            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')

            return [img], [mask], [[0, sub_h, 0, sub_w, sub_h, sub_w]]

#------
# torchvision.transforms.Compose(transforms)将参数列表的预处理依次运行一遍

# torchvision.transforms.CenterCrop(size)中心截取一块

# torchvision.transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0)亮度、对比度和颜色的变化

# torchvision.transforms.FiveCrop(size)四个角以及中间各截取一块

# torchvision.transforms.Grayscale(num_output_channels=1)灰度化

# torchvision.transforms.Pad(padding,fill=0,padding_mode='constant')周围填充，padding_mode决定了是全填充0还是填充和最外圈一样的，具体查文档

# torchvision.transforms.RandomAffine(degrees,translate=None,scale=None,shear=None,resample=0,fillcolor=0)随机仿射变换

# torchvision.transforms.RandomApply(transforms,p=0.5)对于列表中每一个都有p的概率执行

# torchvision.transforms.RandomCrop(size,padding=None,pad_if_needed=False,fill=0,padding_mode='constant')随机位置截取

# torchvision.transforms.RandomGrayscale(p=0.1)随机灰度化

# torchvision.transforms.RandomHorizontalFlip(p=0.5)随机进行水平翻转

# torchvision.transforms.RandomPerspective(distortion_scale=0.5,p=0.5,interpolation=2,fill=0)随机进行透视变换

# torchvision.transforms.RandomResizedCrop(size,scale=(0.08,1.0),ratio=(0.75,1.3333333333333333),interpolation=2)随机缩放并截取

# torchvision.transforms.RandomRotation(degrees,resample=False,expand=False,center=None,fill=None)随机旋转正负degrees度

# torchvision.transforms.RandomVerticalFlip(p=0.5)随机进行垂直翻转

# torchvision.transforms.Resize(size,interpolation=2)缩放

# torchvision.transforms.TenCrop(size,vertical_flip=False)对图片进行上下左右以及中心裁剪，然后全部翻转得到10张图片，vertical_flip决定了是上下翻转还是左右翻转

# torchvision.transforms.GaussianBlur(kernel_size,sigma=(0.1,2.0))高斯平滑

# torchvision.transforms.RandomChoice(transforms)给定的列表随机选择一个执行

# torchvision.transforms.RandomOrder(transforms)给定的列表随机打乱顺序后执行

# torchvision.transforms.LinearTransformation(transformation_matrix,mean_vector)线性变换

# torchvision.transforms.Normalize(mean,std,inplace=False)标准化，mean表示整个数据集的均值，std表示整个数据集的标准差

# torchvision.transforms.RandomErasing(p=0.5,scale=(0.02,0.33),ratio=(0.3,3.3),value=0,inplace=False)随机擦除

# torchvision.transforms.ConvertImageDtype(dtype: torch.dtype)转换数据类型

# torchvision.transforms.ToPILImage(mode=None)转PIL数组

# torchvision.transforms.ToTensor 转tensor

# torchvision.transforms.Lambda(lambd)以给定的函数方式进行


# transfrom之五种Crop裁剪的方法
# 随机裁剪
# 中心裁剪
# 随机长宽比裁剪
# 上下左右中心裁剪
# 上下左右中心裁剪后翻转
# 总共分成四大类：

# 剪裁Crop
# 翻转旋转Flip and Rotation
# 图像变换
# 对transform的操作
# Crop
# 随机裁剪
# class torchvision.transforms.RandomCrop(size,padding=None,pad_if_need=False,fill=0,padding_mode='constant')

# 依据给定size随机剪裁：
# size:要么是（h，w），若是一个int，就是（size，size）
# padding：填充多少pixel。要是一个数，就是上下左右都填充这么多；要是两个数，第一个数就是左右扩充多少，第二个数是上下扩充多少，要是四个数就是 左上右下
# fill：填充的值是什么（仅当填充模式是constant的时候有用）。如果是一个数字，就表示各个通道都填充这个数组，如果是3元tuple，就是RGB三个通道分别填充多少
# padding_mode:填充模式，有四种模式。
# 1，constant，常量
# 2，edge。按照图片边缘像素值填充
# 3，reflect和sysmetric还不了解

# 中心裁剪
# class torchvision.transforms.CenterCrop(size)
# 依据跟定的size，从中心进行裁剪

# 随机长宽比裁剪
# class torchvision.transforms.RandomResizedCrop(size,scale=(0.08,1.0),ratio=(0.75,1.33),interpolation=2)
# 功能：随机大小，随机长宽裁剪原始照片，最后将照片resize到设定好的size
# 参数：
# size：输出的分辨率，就是输出的大小
# scale：随机剪裁的大小区间，上体来说，crop出来的图片会在0.08倍到1倍之间
# ratio：随机长宽比设置
# interpolation：插值的方法。

# 上下左右中心裁剪
# class torchvision.transforms.FiveCrop(size)
# 功能：对图片进行上下左右以及中心的裁剪，获得五张图片，返回一个4D的tensor。

# 上下左右中心裁剪后翻转
# class torchvision.transforms.TenCrop(size,vertical_flip=False)
# 功能：对图片进行上下左右以及中心裁剪，然后全部翻转（水平或者垂直，总共获得十张图片）
