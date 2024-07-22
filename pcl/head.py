"""
@Description : YoloV8 Head for Contrastive Learning
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/21 16:08:33
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f


class Yolov8Head(nn.Module):
    def __init__(self, channels):
        super(Yolov8Head, self).__init__()
        # channels is a list of input channels for each stage
        _, c2, c3, c4 = channels
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_1 = C2f(c3 + c4, 512)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_2 = C2f(512 + c2, 256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.c2f_3 = C2f(256 + c2, 512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.c2f_4 = C2f(512 + c4, 1024)

    def forward(self, input_list):
        assert len(input_list) == 4
        _, x3, x2, x1 = input_list
        x = self.up1(x1)  # torch.Size([1, 2048, 40, 40])
        x = torch.cat((x, x2), dim=1)  # torch.Size([1, 3072, 40, 40])
        x = self.c2f_1(x)  # torch.Size([1, 512, 40, 40])
        hidden_x = x

        x = self.up2(x)  # torch.Size([1, 512, 80, 80])
        x = torch.cat((x, x3), dim=1)  # torch.Size([1, 1024, 80, 80])
        x = self.c2f_2(x)  # torch.Size([1, 256, 80, 80])
        p1 = x

        x = self.conv2(x)  # torch.Size([1, 256, 40, 40])
        x = torch.cat((x, hidden_x), dim=1)  # torch.Size([1, 768, 40, 40])
        x = self.c2f_3(x)  # torch.Size([1, 512, 40, 40])
        p2 = x

        x = self.conv3(x)  # torch.Size([1, 512, 20, 20])
        x = torch.cat((x, x1), dim=1)  # torch.Size([1, 2560, 20, 20])
        x = self.c2f_4(x)
        p3 = x  # torch.Size([1, 1024, 20, 20])

        return [p1, p2, p3]
