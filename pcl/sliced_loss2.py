'''
Author: Estellasy 115784832+Estellasy@users.noreply.github.com
Date: 2024-08-13 14:18:32
LastEditors: Estellasy 115784832+Estellasy@users.noreply.github.com
LastEditTime: 2024-08-13 19:42:56
FilePath: /PCL/pcl/sliced_loss2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch.nn import functional as F

SCALING_FACTOR = 1
# 通过在多个随机方向上对数据进行投影，计算一维分布之间的WD来估计多维分布的WD
# 计算效率高
# 引入额外的参数，如方向的数量和投影方向的本身，来适应特定任务的需求
def sliced_loss(x, y):
    # x: [1, 1024, 7, 7]
    # y: [1, 1024, 7, 7]
    b, dim, h, w = x.shape
    n = h*w
    x = x.view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR)
    y = y.view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR)
    # print(x.shape, y.shape)
    # sample more random directions
    # 采样更多的随机方向
    Ndirection = 10 * dim  # 增加方向数量
    directions = torch.randn(Ndirection, dim)
    # 归一化
    directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
    # project activations over random directions
    projected_x = torch.einsum('bdn,md->bmn', x, directions)
    projected_y = torch.einsum('bdn,md->bmn', y, directions)
    # sort the projections
    sorted_activations_x = torch.sort(projected_x, dim=2)[0]
    sorted_activations_y = torch.sort(projected_y, dim=2)[0]
    # 使用调整后的余弦相似度
    adjusted_cosine_similarity = F.cosine_similarity(sorted_activations_x, sorted_activations_y, dim=2)
    # 调整余弦相似度，使其更接近于 0
    adjusted_cosine_similarity = 1 - adjusted_cosine_similarity
    # adjusted_cosine_similarity = (adjusted_cosine_similarity + 1) / 2
    # 计算平均调整后的余弦相似度
    return torch.mean(adjusted_cosine_similarity)


if __name__ == "__main__":
    x = torch.randn((1, 1024, 7, 7))
    y = torch.randn((1, 1024, 7, 7))
    print(sliced_loss(x, y))
    print(sliced_loss(x, x))
    print(sliced_loss(y, y*0.8))