# 计算两个tensor之间的emd距离
import torch
import torch.nn.functional as F

class SinkhornDistance(torch.nn.Module):
    def __init__(self, eps=1e-3, max_iter=100, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # 扁平化空间维度
        batch_size, num_points, height, width = x.size()
        x = x.view(batch_size, num_points, -1)
        y = y.view(batch_size, num_points, -1)

        # 归一化特征向量到概率分布
        x = F.softmax(x, dim=-1)
        y = F.softmax(y, dim=-1)

        # 计算 Euclidean 距离作为成本矩阵
        cost_matrix = torch.cdist(x, y, p=2)

        # 初始分布
        u = torch.ones(batch_size, num_points).to(x.device) / num_points
        v = torch.ones(batch_size, num_points).to(x.device) / num_points

        # Sinkhorn 迭代
        for _ in range(self.max_iter):
            u = self.eps / (cost_matrix.bmm(v.unsqueeze(-1)).squeeze(-1) + self.eps)
            v = self.eps / (cost_matrix.transpose(1, 2).bmm(u.unsqueeze(-1)).squeeze(-1) + self.eps)

        # 计算 Wasserstein 距离
        distance = (u.unsqueeze(-1) * cost_matrix * v.unsqueeze(-2)).sum(dim=(1, 2))

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
print(distance) # tensor(1.0227)
similarity = torch.exp(-distance)
print(similarity)

distance2 = sinkhorn(x1, x1)
print(distance2)    # tensor(1.0227)

x3 = torch.randn(1, 1024, 7, 7)
distance3 = sinkhorn(x1, x3)
print(distance3)    # tensor(1.0227)
