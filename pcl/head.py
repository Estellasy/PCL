"""
@Description : YoloV8 Head for Contrastive Learning
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/21 16:08:33
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f

class MlpHead(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(MlpHead, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # mlp: fc-relu-fc
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def forward(self, x):
        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))
        return avgpooled_x


class Yolov8HeadBak(nn.Module):
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

        self.conv_fuse = nn.Conv2d(1792, 1024, kernel_size=1, stride=1, padding=0)
        self.bn_fuse = nn.BatchNorm2d(1024)
        self.relu_fuse = nn.ReLU(inplace=True)

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

        # 特征融合
        # Fusion process
        p1 = nn.functional.adaptive_avg_pool2d(p1, (7, 7))  # torch.Size([1, 256, 20, 20])
        p2 = nn.functional.adaptive_avg_pool2d(p2, (7, 7))  # torch.Size([1, 512, 20, 20])
        p3 = p3  # torch.Size([1, 1024, 20, 20])


        fused = torch.cat((p1, p2, p3), dim=1)  # torch.Size([1, 1792, 20, 20])
        fused = self.conv_fuse(fused)  # torch.Size([1, 1024, 20, 20])
        fused = self.bn_fuse(fused)
        fused = self.relu_fuse(fused)

        return fused
        # return [p1, p2, p3]


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

        ch_1, ch_2, ch_int, r_2, ch_out = 256, 1024, 512, 4, 512
        self.fuse = MultiScaleFusion(ch_1, ch_2, r_2, ch_int, ch_out)

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

        # 特征融合
        # Fusion process
        return self.fuse(p1, p3, p2)

"""
@Description : Multi-scale Fusion
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/21 16:08:33

Multi-scale Progressive Fusion Network
Adaptive 多尺度渐进融合层

给定道路图像，坑洼可以具有不同的形状和尺度。我们可以通过一系列的卷积和池化操作获得顶层的特征图。
虽然特征图具有丰富的语义信息，但其分辨率不足以提供准确的语义预测。不幸的是，直接结合低级特征图只能带来非常有限的改进。
为了克服这个缺点，研究者设计了一个有效的特征融合模块。一种方法是直接结合低级特征图

Low-Level特征图

该模块有助于在不引入额外参数的情况下保留更多细节。
此外，采用Deeplabv3中使用的ASPP模块来收集顶层特征图中的上下文信息。
然后，采用CAM重新加权不同通道中的特征图。它可以突出一些特征，从而产生更好的语义预测。
最后，将不同级别的特征图输入到MSFFM中，以提高坑洼轮廓附近的分割性能。

将不同级别的特征图输入到MSFFM中，提高轮廓信息。但是高级特征我们不清楚他的语义。

在YOLOv8 Head之后，经过一个特征金字塔，从金字塔层学习尺度特定的知识。并且减少特征冗余。
浅层特征图，纹理信息。
精细融合模块，进一步整合不同尺度的相关信息。
使用通道注意力机制，从

高分辨率金字塔层的表示由先前的输出以及所有低分辨率金字塔层引导。
接下来是精细融合模块（FFM）以进一步整合来自不同尺度的相关信息。
通过使用通道注意机制，网络不仅可以从所有前面的金字塔层中有区别地学习尺度特定的知识，而且还有效地减少了特征冗余。此外，多个FFM可以级联以形成渐进式多尺度融合。最后，附加一个重建模块（RM）来聚合分别从 CFM 和 FFM 中提取的粗雨和细雨信息，以学习残差雨图像，这是真实雨纹分布的近似值。

从哪里学到什么信息

为什么学到这个信息

并且特征金字塔网络通过将高级语义特征与空间细节合并来丰富特征表示。

数据集质量不高

引入了一个YOLOv8 Head实现特征提取和对齐。

因为数据质量不高，所以我们设计了合理的方式来融合。

multi-scale progressive fusion

对于低级尺度的特征：blabla

对于中高尺度的特征：blabla 进行了第一次融合

对于低级尺度特征和中高尺度特征：再次进行融合

此外，我们引入了一种深度监督策略来指导模型融合多尺度特征信息。它进一步使 SMTF 能够有效地跨层传播特征信息，以保留更多的输入空间信息并减轻信息衰减。

BGF-YOLO 包含一种注意力机制，可以更加关注重要特征，并且特征金字塔网络通过将高级语义特征与空间细节合并来丰富特征表示。此外，我们研究了不同的注意力机制和特征融合、检测头架构对脑肿瘤检测准确率的影响。

我们修改了 YOLOv8 中的 FPN-PANet 结构，通过加强网络的多路径融合来实现不同层之间的多级特征融合。受到 GFPN 和基于重新参数化的 GFPN 的 DAMO-YOLO [28] 的启发，我们利用 Cross Stage Partial DenseNet (CSP) [26] 添加跳过连接，并通过替换 C2f（无捷径）并与 Conv 结合，同时在各种空间尺度和非相邻的潜在语义级别之间共享密集信息。这使模型能够在颈部以同等重要性处理高级语义信息和低级空间信息。

https://blog.csdn.net/athrunsunny/article/details/128920297
DAMO yolo在提出前阿里就推出了一个叫GIRAFFEDET的模型，主推轻backbone重neck的网络结构，使得网络更关注于高分辨率特征图中空间信息和低分辨率特征图中语义信息的信息交互。同时这个设计范式允许检测网络在网络早期阶段就以相同优先级处理高层语义信息和低层空间信息，使其在检测任务上更加有效。

使得网络关注backbone中不同层级的信息。在得到高分辨率空间信息和低分辨率语义信息后，进行融合

允许检测网络在网络早期阶段就以相同优先级处理高层语义信息和低层空间信息，使其在检测任务上更加有效。

如何adaptive去选择高层语义信息和底层空间信息 dense层面

参考NAFNet的模块，我们拆分为通道注意力和注意力，
"""


class MultiScaleFusion(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out):
        super(MultiScaleFusion, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(ch_int // 2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        self.gelu = nn.GELU()
        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)  # 多层感知机用于非线性映射

    def forward(self, l, g, f):
        # l: torch.Size([4, 256, 28, 28])
        # g: torch.Size([4, 512, 14, 14])
        # f: torch.Size([4, 1024, 7, 7])
        # print(l.shape)
        # print(g.shape)
        # print(f.shape)
        W_local = self.W_l(l)
        W_local = self.Avg(W_local)
        # print("W_local: ", W_local.shape)   # [4, 512, 14, 14]
        W_global = self.W_g(g)
        W_global = self.upsample(W_global)
        # print("W_global: ", W_global.shape) # [1, 512, 14, 14]
        if f is not None:       # 额外特征
            # W_f = self.Updim(f) # 上采样
            W_f = f
            # print("W_f: ", W_f.shape)   # W_f:  torch.Size([1, 512, 14, 14])
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)    # 在通道维度进行拼接
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)  # 卷积
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        # print("X_f.shape", X_f.shape)   # X_f.shape torch.Size([1, 512, 14, 14])

        # spatial attention for ConvNeXt branch 空间注意力
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)   # 最大池化
        avg_result = torch.mean(l, dim=1, keepdim=True)     # 平均池化
        result = torch.cat([max_result, avg_result], 1)     # sigmoid得到加权的l
        l = self.spatial(result)
        l = self.sigmoid(l) * l_jump                        # 空间注意力

        # print("l.shape:", l.shape)  # [1, 256, 80, 80]

        # channel attention for transformer branch
        g_jump = g                                          # 通道注意力
        max_result = self.maxpool(g)
        avg_result = self.avgpool(g)
        max_out = self.se(max_result)
        # print("max_out.shape:", max_out.shape)
        avg_out = self.se(avg_result)
        # print("avg_out.shape:", avg_out.shape)
        g = self.sigmoid(max_out + avg_out) * g_jump        # 通道注意力
        # print("g.shape:", g.shape)          # [1, 1024, 20, 20]
        g = self.upsample(g)
        # print("g.shape:", g.shape)          # [1, 1024, 40, 40]

        # ADJUST THE SIZE OF L AND G TO MATCH X_F ##
        _, _, H, W = X_f.shape
        l_resized = nn.functional.adaptive_avg_pool2d(l, (H, W))
        # print("l_resized.shape:", l_resized.shape)

        fuse = torch.cat([g, l_resized, X_f], 1)                    # 特征拼接
        # print("fuse.shape:", fuse.shape)  # [1, 1280, 40, 40]
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)                          # 残差连接
        fuse = shortcut
        return fuse


class MultiScaleFusionTest(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out):
        super(MultiScaleFusion, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(ch_int // 2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        self.gelu = nn.GELU()
        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)  # 多层感知机用于非线性映射

    def forward(self, l, g, f):
        W_local = self.W_l(l)
        W_local = self.Avg(W_local)
        # print("W_local: ", W_local.shape)   # [1, 512, 40, 40]
        W_global = self.W_g(g)
        W_global = self.upsample(W_global)
        # print("W_global: ", W_global.shape) # [1, 512, 40, 40]
        if f is not None:       # 额外特征
            W_f = self.Updim(f) # 上采样
            # print("W_f: ", W_f.shape)   # [1, 512, 40, 40]
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)    # 在通道维度进行拼接
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)  # 卷积
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        # print("X_f.shape", X_f.shape)   # [1, 512, 40, 40]

        # spatial attention for ConvNeXt branch 空间注意力
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)   # 最大池化
        avg_result = torch.mean(l, dim=1, keepdim=True)     # 平均池化
        result = torch.cat([max_result, avg_result], 1)     # sigmoid得到加权的l
        l = self.spatial(result)
        l = self.sigmoid(l) * l_jump                        # 空间注意力

        # print("l.shape:", l.shape)  # [1, 256, 80, 80]

        # channel attetion for transformer branch
        g_jump = g                                          # 通道注意力
        max_result = self.maxpool(g)
        avg_result = self.avgpool(g)
        max_out = self.se(max_result)
        # print("max_out.shape:", max_out.shape)
        avg_out = self.se(avg_result)
        # print("avg_out.shape:", avg_out.shape)
        g = self.sigmoid(max_out + avg_out) * g_jump        # 通道注意力
        # print("g.shape:", g.shape)          # [1, 1024, 20, 20]
        g = self.upsample(g)
        # print("g.shape:", g.shape)          # [1, 1024, 40, 40]

        ## ADJUST THE SIZE OF L AND G TO MATCH X_F ##
        _, _, H, W = X_f.shape
        l_resized = nn.functional.adaptive_avg_pool2d(l, (H, W))
        # print("l_resized.shape:", l_resized.shape)

        fuse = torch.cat([g, l_resized, X_f], 1)                    # 特征拼接
        # print("fuse.shape:", fuse.shape)  # [1, 1280, 40, 40]
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)                          # 残差连接
        fuse = shortcut
        return fuse


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class IRMLP(nn.Module):
    """
    Inverted Residual Multi-Layer Perceptron
    conv1 进行逐通道卷积 保留输入通道不变 增加非线性
    conv2 拓展通道数 进一步特征特征
    conv3 压缩通道数 规范化
    gelu 增加非线性
    bn1 批归一化
    """
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


def test_multiscale_fusion():
    # 定义输入的通道数和尺寸
    ch_1 = 256
    ch_2 = 1024
    ch_int = 512
    r_2 = 4
    ch_out = 512

    h_1 = 28
    h_2 = 7

    # 创建 MultiScaleFusion 实例
    model = MultiScaleFusion(ch_1, ch_2, r_2, ch_int, ch_out)

    # 打印模型结构
    # print(model)

    # 创建示例输入
    batch_size = 1

    l = torch.randn(batch_size, ch_1, h_1, h_1)  # 局部特征输入
    # print("l.shape", l.shape)
    g = torch.randn(batch_size, ch_2, h_2, h_2)  # 全局特征输入
    # print("g.shape", g.shape)
    f = torch.randn(batch_size, ch_int, h_1 // 2, h_1 // 2)  # 额外特征输入
    # print("f.shape", f.shape)

    # 前向传播
    output = model(l, g, f)

    # 打印输出形状
    # print(f"输出形状: {output.shape}")


def test_yolohead():
    pass


if __name__ == '__main__':
    test_multiscale_fusion()

