import torch
import torch.nn.functional as F

SCALING_FACTOR = 1


# 通过在多个随机方向上对数据进行投影，计算一维分布之间的WD来估计多维分布的WD
# 计算效率高
# 引入额外的参数，如方向的数量和投影方向的本身，来适应特定任务的需求
def sliced_loss(x, y):
    # x: [1, 1024, 7, 7]
    # y: [1, 1024, 7, 7]
    b, dim, h, w = x.shape
    n = h * w
    x = x.view(b, dim, n).repeat(1, 1, SCALING_FACTOR * SCALING_FACTOR)
    y = y.view(b, dim, n).repeat(1, 1, SCALING_FACTOR * SCALING_FACTOR)
    # print(x.shape, y.shape)
    # sample random directions
    # 采样dim个随机方向
    Ndirection = dim
    directions = torch.randn(Ndirection, dim, device=x.device)
    # 归一化
    directions = directions / torch.sqrt(torch.sum(directions ** 2, dim=1, keepdim=True))
    # project activations over random directions
    projected_x = torch.einsum('bdn,md->bmn', x, directions)
    projected_y = torch.einsum('bdn,md->bmn', y, directions)
    # sort the projections
    sorted_activations_x = torch.sort(projected_x, dim=2)[0]
    sorted_activations_y = torch.sort(projected_y, dim=2)[0]
    # print(sorted_activations_x.shape, sorted_activations_y.shape)
    cosine_similarity = F.cosine_similarity(sorted_activations_x, sorted_activations_y, dim=2)

    # L2 over sorted lists
    return torch.mean((sorted_activations_x - sorted_activations_y) ** 2) * 1e5
    # return 1 - torch.mean(cosine_similarity)


if __name__ == "__main__":
    x = torch.randn((8, 1024, 7, 7), requires_grad=True)
    y = torch.randn((8, 1024, 7, 7), requires_grad=True)
    print(sliced_loss(x, y))
    print(sliced_loss(x, x))
    print(sliced_loss(y, x*0.5))
