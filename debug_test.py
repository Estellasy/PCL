import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

# 生成随机特征图
x1 = torch.randn(1, 1024, 7, 7)
x2 = torch.randn(1, 1024, 7, 7)

# 展平特征图并归一化
x1_flat = x1.view(1, 1024, -1)
x2_flat = x2.view(1, 1024, -1)

# 转换为概率分布
x1_prob = F.softmax(x1_flat, dim=-1)
x2_prob = F.softmax(x2_flat, dim=-1)

# 计算 EMD 距离
emd_distance = wasserstein_distance(x1_prob.cpu().numpy().flatten(), x2_prob.cpu().numpy().flatten())
print(emd_distance)