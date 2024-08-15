import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        
    def forward(self, pos, neg):
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        return self.criterion(logits, labels)
    
    


class DenseContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.01, contrast_mode='all', base_temperature=0.01):
        super(DenseContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, feature1, feature2, labels=None, mask=None):
        device = feature1.device
        feature1 = feature1.view(feature1.shape[0], feature1.shape[1], -1)
        feature2 = feature2.view(feature2.shape[0], feature2.shape[1], -1)

        batch_size = feature1.shape[0]  # 32
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # [32, 32]
            # print(mask.shape)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)  # [32, 1]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float()  # [32, 32]
        else:
            mask = mask.float()

        contrast_count = 2
        contrast_feature = torch.cat([feature1, feature2], dim=0)  # [64, 128, 49]
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # [32, 128, 49]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # [64, 128, 49]
            anchor_count = contrast_count  # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        
        # print("anchor_feature:", anchor_feature.shape)
        # compute logits using Frobenius norm
        def frobenius_norm(tensor1, tensor2):
            return torch.sum((tensor1 - tensor2) ** 2, dim=[2, 3])
        
        # 计算 anchor_dot_contrast
        anchor_dot_contrast = -frobenius_norm(
            anchor_feature.unsqueeze(1),  # [64, 1, 128, 49]
            contrast_feature.unsqueeze(0)  # [1, 64, 128, 49]
        ) * self.temperature  # 结果形状为 [64, 64]
        
        # print("anchor_dot_contrast:", anchor_dot_contrast)

        # 数值稳定性处理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 数值稳定性处理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 处理掩码和计算 log_prob
        mask = mask.repeat(anchor_count, contrast_count)  # [64, 64]
        logits_mask = torch.ones_like(logits).to(device)  # [64, 64]
        logits_mask = torch.scatter(
            logits_mask,
            1,
            torch.arange(logits.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # [64, 64]

        exp_logits = torch.exp(logits) * logits_mask  # [64, 64]
        # 添加一个小常数以避免数值问题
        safe_exp_logits_sum = exp_logits.sum(1, keepdim=True) + 1e-10
        log_prob = logits - torch.log(safe_exp_logits_sum)  # [64, 64]

        # 计算平均对数似然损失
        mask_pos_pairs = mask.sum(1)  # [64]
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=mask_pos_pairs.dtype, device=device), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs  # [64]

        # 计算最终损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    
if __name__ == "__main__":
    denseloss = DenseContrastiveLoss()
    feature1 = torch.randn(1, 1024, 7, 7)
    feature2 = torch.randn(1, 1024, 7, 7)
    print(denseloss(feature1, feature2))
    print(denseloss(feature1, feature1))
    print(denseloss(feature2, feature2))