"""
@Description :
@Author      : siyiren1@foxmail.com
@Time        : 2024/08/10 21:43:15
"""
import torch
import torch.nn.functional as F
import cv2


def get_similarity_map(x, y, metric='cosine'):
    way = x.shape[0]
    num_y = y.shape[0]
    y = y.view(y.shape[0], y.shape[1], -1)
    x = x.view(x.shape[0], x.shape[1], -1)

    x = x.unsqueeze(0).repeat([num_y, 1, 1, 1])
    y = y.unsqueeze(1).repeat([1, way, 1, 1])
    x = x.permute(0, 1, 3, 2)
    y = y.permute(0, 1, 3, 2)
    feature_size = x.shape[-2]

    if metric == 'cosine':
        x = x.unsqueeze(-3)
        y = y.unsqueeze(-2)
        y = y.repeat(1, 1, 1, feature_size, 1)
        similarity_map = F.cosine_similarity(x, y, dim=-1)
    else:
        x = x.unsqueeze(-3)
        y = y.unsqueeze(-2)
        y = y.repeat(1, 1, 1, feature_size, 1)
        similarity_map = (x - y).pow(2).sum(-1)
        similarity_map = 1 - similarity_map

    return similarity_map


def emd_inference(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, _ = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost


def compute_emd_loss(x, y):
    similarity_map = get_similarity_map(x, y)

    num_x = similarity_map.shape[0]
    num_y = similarity_map.shape[1]
    num_node = x.shape[-1]

    total_cost = 0
    for i in range(num_x):
        for j in range(num_y):
            cost = emd_inference(1 - similarity_map[i, j, :, :], x[i, j, :], y[j, i, :])
            total_cost += cost

    # Average cost over all pairs
    emd_loss = total_cost / (num_x * num_y)
    return emd_loss


if __name__ == "__main__":
    x = torch.randn((1, 1024, 7, 7))
    y = torch.randn((1, 1024, 7, 7))
    emd_loss = compute_emd_loss(x, y)
    print(f'EMD Loss: {emd_loss}')

    emd_loss = compute_emd_loss(x, x)
    print(f'EMD Loss: {emd_loss}')

    emd_loss = compute_emd_loss(y, y)
    print(f'EMD Loss: {emd_loss}')