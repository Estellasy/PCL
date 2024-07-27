"""
@Description : dataloader of self-supervised train for defect detectio
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/22 00:20:08
"""

"""
https://www.cnblogs.com/lvdongjie/p/14056273.html
CutPaste: https://github.com/LilitYolyan/CutPaste
MixUp: https://github.com/hongyi-zhang/mixup
"""

import torch
from torchvision import datasets, transforms
from PIL import Image, ImageFilter
import numpy as np
import random
import os


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index


class RandomImageFolderInstance(ImageFolderInstance):
    """继承自ImageFolderInstance
    MixUp和CutMix混合增强
    """

    def __init__(self, root, transform=None, cutpaste_transform=None, mixup_alpha=1.0):
        super(RandomImageFolderInstance, self).__init__(root, transform)
        self.cutpaste_transform = cutpaste_transform
        self.mixup_alpha = mixup_alpha

    def __getitem__(self, index):
        # 读取主样本
        x_i, _ = super(RandomImageFolderInstance, self).__getitem__(index)
        # 随机选择两个样本 x_a 和 x_b
        indices = list(range(len(self.samples)))
        indices.remove(index)
        x_a_index = random.choice(indices)
        indices.remove(x_a_index)
        x_b_index = random.choice(indices)

        x_a, _ = super(RandomImageFolderInstance, self).__getitem__(x_a_index)
        x_b, _ = super(RandomImageFolderInstance, self).__getitem__(x_b_index)

        # 应用CutPaste增强
        if self.cutpaste_transform is not None:
            x_i = self.cutpaste_transform(x_i, x_a)
        # 应用MixUp增强
        # 应用 MixUp 增强
        output = []
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            # print(type(x_i), type(x_b))
            # print(len(x_i))
            for (i, b) in zip(x_i, x_b):
                x = lam * i + (1 - lam) * b
                output.append(x)

        return output, index


class CutPasteTransform:
    def __call__(self, x_i, x_a):
        # 简单的二值分割，获取前景和背景
        x_i_gray = transforms.Grayscale()(x_i)
        x_a_gray = transforms.Grayscale()(x_a)

        threshold_i = x_i_gray.point(lambda p: p > 128 and 255)
        threshold_a = x_a_gray.point(lambda p: p > 128 and 255)

        x_i_np = np.array(x_i)
        x_a_np = np.array(x_a)

        # 随机选择前景区域的一个小块
        foreground_indices = np.where(threshold_a)
        if len(foreground_indices[0]) == 0:
            return x_i  # 如果没有前景区域，返回原图

        random_index = random.choice(range(len(foreground_indices[0])))
        start_row, start_col = (
            foreground_indices[0][random_index],
            foreground_indices[1][random_index],
        )

        patch_size = (32, 32)
        patch = x_a_np[
            start_row : start_row + patch_size[0],
            start_col : start_col + patch_size[1],
            :,
        ]

        # 粘贴到 x_i 的前景区域
        x_i_np[
            start_row : start_row + patch_size[0],
            start_col : start_col + patch_size[1],
            :,
        ] = patch

        return Image.fromarray(x_i_np)


class RandomImageFolderInstance4Test(RandomImageFolderInstance):
    """相比之下添加了保存图片示例的代码"""

    def __init__(
        self,
        root,
        transform=None,
        cutpaste_transform=None,
        mixup_alpha=1.0,
        save_dir="enhanced_images",
    ):
        super(RandomImageFolderInstance4Test, self).__init__(root, transform)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __getitem__(self, index):
        # 读取主样本
        x_i, _ = super(RandomImageFolderInstance, self).__getitem__(index)

        # 随机选择两个样本 x_a 和 x_b
        indices = list(range(len(self.samples)))
        indices.remove(index)
        x_a_index = random.choice(indices)
        indices.remove(x_a_index)
        x_b_index = random.choice(indices)

        x_a, _ = super(RandomImageFolderInstance, self).__getitem__(x_a_index)
        x_b, _ = super(RandomImageFolderInstance, self).__getitem__(x_b_index)

        # 保存原始图像
        x_i.save(os.path.join(self.save_dir, f"original_{index}.png"))
        x_a.save(os.path.join(self.save_dir, f"x_a_{x_a_index}.png"))
        x_b.save(os.path.join(self.save_dir, f"x_b_{x_b_index}.png"))

        # 应用 CutPaste 增强
        if self.cutpaste_transform is not None:
            x_i = self.cutpaste_transform(x_i, x_a)

        # 保存 CutPaste 增强后的图像
        x_i.save(os.path.join(self.save_dir, f"cutpaste_{index}.png"))

        # 应用 MixUp 增强
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x_i = np.array(x_i) * lam + np.array(x_b) * (1 - lam)
            x_i = Image.fromarray(x_i.astype(np.uint8))

        # 保存 MixUp 增强后的图像
        x_i.save(os.path.join(self.save_dir, f"mixup_{index}.png"))

        return x_i, index
