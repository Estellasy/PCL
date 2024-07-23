"""
@Description :
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/22 00:08:11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornDistance(torch.nn.Module):
    def __init__(self, eps, max_iter, reduction="mean"):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # 扁平化空间维度
        batch_size, num_points, height, width = x.size()
        x = x.view(batch_size, num_points, -1)
        y = y.view(batch_size, num_points, -1)

        # 归一化特征向量
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # 计算余弦相似度矩阵并归一化
        cost_matrix = 1 - F.cosine_similarity(x.unsqueeze(2), y.unsqueeze(1), dim=-1)
        cost_matrix = cost_matrix / cost_matrix.max()  # 归一化到 [0, 1] 范围内

        # 初始分布
        u = torch.ones(batch_size, num_points).to(x.device) / num_points
        v = torch.ones(batch_size, num_points).to(x.device) / num_points

        # Sinkhorn 迭代
        for _ in range(self.max_iter):
            u = 1.0 / (cost_matrix.bmm(v.unsqueeze(-1)).squeeze(-1) + self.eps)
            v = 1.0 / (
                cost_matrix.transpose(1, 2).bmm(u.unsqueeze(-1)).squeeze(-1) + self.eps
            )

        # 计算Wasserstein距离
        distance = (u.unsqueeze(-1) * cost_matrix * v.unsqueeze(-2)).sum(dim=(1, 2))

        if self.reduction == "mean":
            distance = distance.mean()
        elif self.reduction == "sum":
            distance = distance.sum()

        return distance


class WassersteinLoss(nn.Module):
    def __init__(self, eps=0.1, max_iter=100, reduction="mean"):
        super(WassersteinLoss, self).__init__()
        self.sinkhorn_distance = SinkhornDistance(eps, max_iter, reduction)

    def forward(self, pred, target):
        return self.sinkhorn_distance(pred, target)


class EMDLoss(nn.Module):
    def __init__(self, eps=0.1, max_iter=100, reduction="mean"):
        super().__init__()
        self.loss = 0.0
        self.wasserstein_loss = WassersteinLoss(eps, max_iter, reduction)

    def forward(self, features1, features2):
        assert len(features1) == len(features2)
        for x1, x2 in zip(features1, features2):
            self.loss += self.wasserstein_loss(x1, x2)
        return self.loss


if __name__ == "__main__":
    pred = torch.randn(4, 8, 32, 32)
    target = torch.randn(4, 8, 32, 32)

    emd_loss = EMDLoss(eps=0.1, max_iter=100)
    loss = emd_loss([pred], [target])
    print("Wasserstein Loss:", loss)
