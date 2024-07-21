"""
@Description :
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/22 00:08:11
"""


"""
An unofficial PyTorch implementation of "A Sliced Wasserstein Loss for Neural Texture Synthesis" paper [CVPR 2021].
https://github.com/xchhuang/pytorch_sliced_wasserstein_loss/blob/main/pytorch/loss_fn.py
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class EMD(nn.Module):
    def __init__(self, device, layers, repeat_rate):
        super().__init__()
        # Number of directions
        