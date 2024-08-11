import torch
import torch.nn as nn
import torch.nn.functional as F

class SinkhornSim(torch.nn.Module):
    def __init__(self, eps=1e-3, max_iter=100, reduction='sum'):
        super(SinkhornSim, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        
    def forward(self, x, y): # 在这里将输入的维度更改为[B, C, N]
        # 扁平化空间维度
        B, C, num_points = x.size()
        x = x.view(B, num_points, C) 
        y = y.view(B, num_points, C) 

        # 转换为概率分布
        x = F.softmax(x, dim=-1)
        y = F.softmax(y, dim=-1)

        # 计算成本矩阵 (Euclidean 距离)
        cost_matrix = torch.cdist(x, y, p=2) ** 2  
        # cost_matrix = torch.cdist(x, y, p=2)  

        # Sinkhorn-Knopp 迭代的初始化
        K = torch.exp(-cost_matrix / self.eps)
        u = torch.ones(B, num_points).to(x.device) / num_points
        v = torch.ones(B, num_points).to(y.device) / num_points

        #Sinkhorn 迭代 
        for _ in range(self.max_iter):
            u = 1.0 / (K.bmm(v.unsqueeze(-1)).squeeze(-1) + 1e-8)
            v = 1.0 / (K.transpose(1, 2).bmm(u.unsqueeze(-1)).squeeze(-1) + 1e-8)

        # 计算Wasserstein 距离
        transport_plan = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        distance = torch.sum(transport_plan * cost_matrix, dim=(1, 2)) 
       # return torch.exp(-distance)
        return 1-distance / (torch.sum(transport_plan * cost_matrix, dim=(1, 2)) + 1e-9)  
        
class EMD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sinkhorn = SinkhornSim()
        
    def forward(self, x, y):   
        # Cost Matrix
        M = 1 - torch.matmul(x.unsqueeze(2), y.unsqueeze(3)).squeeze(-1).squeeze(-1) # similarity score
        
        # Marginal weights  
        b,c,h,w=x.shape
        r = torch.ones((b,h*w),device=x.device)/h 
        c = torch.ones((b,h*w), device=y.device) / w  
        
        transport_plan, _, _ = self.sinkhorn(M,r,c)  
        
        S = (torch.sum(transport_plan * M)+ 1e-9)  
        
        return 2 - 2 * S  
    
if __name__ == "__main__":
    x1 = torch.randn(1, 1024, 7, 7) 
    x2 = torch.randn(1, 1024, 7, 7)
    emdloss = EMD()
    print(emdloss(x1, x2))
    print(emdloss(x1, x1))