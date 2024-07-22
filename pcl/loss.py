"""
@Description :
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/22 00:08:11
"""


"""
An unofficial PyTorch implementation of "A Sliced Wasserstein Loss for Neural Texture Synthesis" paper [CVPR 2021].
https://github.com/xchhuang/pytorch_sliced_wasserstein_loss/blob/main/pytorch/loss_fn.py
"""

"""
通过SinkhornDistance方法计算Wasserstein距离
https://www.cnblogs.com/wangxiaocvpr/p/11574006.html
https://github.com/Haoqing-Wang/CPNWCP/blob/main/methods/cpn_wcp.py
维度信息还需要仔细看 如何做
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()


    def M(self, C, u, v, eps):
        """Modified cost for logarithmic updates
        C: 成本矩阵
        u/v: 对偶变量 用于迭代更新
        eps: 正则化参数 用于控制平滑程度
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps


    def SinkhornDistance(self, p1, p2, C, iter=5, eps=0.5):
        """通过SinkhornDistance方法计算Wasserstein距离
        p1/p2: 输入的两个概率分布
        """
        u = torch.zeros_like(p1)    # 初始化为全0 Tensor 用于迭代更新
        v = torch.zeros_like(p2)
        for _ in range(iter):
            print("u.shape", u.shape)
            print("v.shape", v.shape)
            print("C.shape", C.shape)
            M_uv = self.M(C, u, v, eps)  # 计算M的结果
            print(M_uv)
            u = eps * (torch.log(p1 + 1e-12) - torch.logsumexp(self.M(C, u, v, eps), dim=-1)) + u
            print("u.shape", u.shape)
            v = eps * (torch.log(p2 + 1e-12) - torch.logsumexp(self.M(C, u, v, eps).transpose(-2, -1), dim=-1)) + v
            print("v.shape", v.shape)

        pi = torch.exp(self.M(C, u, v, eps))    # 联合分布
        return (pi * C).sum((-2, -1)).mean()    # 联合分布和成本矩阵的加权
    

    def _forward(self, x1, x2):
        # 计算Wasserstein距离
        # flatten并norm
        b, c, h, w = x1.shape
        x1 = x1.view(b, c, -1)  # [batch, channels, height * width]
        x2 = x2.view(b, c, -1)

        # feature map -> distribution
        p1 = torch.softmax(x1, dim=-1)  # [batch, channels, height * width]
        p2 = torch.softmax(x2, dim=-1)

        # 成本矩阵计算
        # 欧氏距离
        # c = torch.cdist(p1, p2, p=2)    # [batch, channels, height * width, height * width]

        # 余弦相似度
        cos_sim = F.cosine_similarity(x1.unsqueeze(2), x2.unsqueeze(1), dim=-1)  # [batch, channels, height * width, height * width]
        C = 1 - cos_sim
        C = (C - C.min(dim=-1, keepdim=True)[0]) / (C.max(dim=-1, keepdim=True)[0] - C.min(dim=-1, keepdim=True)[0])

        loss = self.SinkhornDistance(p1, p2, C)
        return loss


    def forward(self, f1, f2, temp=5.):
        assert len(f1) == len(f2)
        loss = 0.
        for (x1, x2) in zip(f1, f2):
            loss += self._forward(x1, x2)
        return loss / len(f1)
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emd_loss = EMDLoss().to(device)
    
    x1 = torch.randn(1, 32, 40, 40).to(device)
    x2 = torch.randn(1, 32, 40, 40).to(device)
    
    loss = emd_loss.forward([x1], [x2])
    print(loss)