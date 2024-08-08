import torch
import torch.nn.functional as F


class SinkhornDistance(torch.nn.Module):
    def __init__(self, eps=1e-3, max_iter=100, reduction='sum'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # 扁平化空间维度
        batch_size, num_points, height, width = x.size()
        x = x.view(batch_size, num_points, -1)
        y = y.view(batch_size, num_points, -1)

        # 转换为概率分布
        x = F.softmax(x, dim=-1)
        y = F.softmax(y, dim=-1)

        # 计算成本矩阵 (Euclidean 距离)
        cost_matrix = torch.cdist(x, y, p=2) ** 2  # 使用平方欧几里得距离
        # cost_matrix = torch.cdist(x, y, p=2)  # 使用欧几里得距离

        # Sinkhorn-Knopp 迭代的初始化
        K = torch.exp(-cost_matrix / self.eps)
        u = torch.ones(batch_size, num_points).to(x.device) / num_points
        v = torch.ones(batch_size, num_points).to(x.device) / num_points

        # Sinkhorn 迭代
        for _ in range(self.max_iter):
            u = 1.0 / (K.bmm(v.unsqueeze(-1)).squeeze(-1) + 1e-8)
            v = 1.0 / (K.transpose(1, 2).bmm(u.unsqueeze(-1)).squeeze(-1) + 1e-8)

        # 计算 Wasserstein 距离
        transport_plan = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        distance = torch.sum(transport_plan * cost_matrix, dim=(1, 2))

        if self.reduction == 'mean':
            distance = distance.mean()
        elif self.reduction == 'sum':
            distance = distance.sum()

        return distance

# 示例使用
x1 = torch.randn(1, 1024, 7, 7)
x2 = torch.randn(1, 1024, 7, 7)
sinkhorn = SinkhornDistance()
distance = sinkhorn(x1, x2)
print(torch.exp(-distance))

distance2 = sinkhorn(x1, x1)
print(torch.exp(-distance2))

# x3 = torch.randn(1, 1024, 7, 7)
x3 = x1 * 1.01
distance3 = sinkhorn(x1, x3)
print(torch.exp(-distance3))

