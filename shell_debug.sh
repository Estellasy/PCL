# Debug
python main_detection.py /home/siyi/project/PCL/data/hg --gpu 0 --workers 4 -a resnet50 --lr 0.03 --batch-size 4 --temperature 0.2 --mlp --aug-plus --cos --multiprocessing-distributed --world-size 1 --rank 0 --exp-dir exp/debug --dist-url 'tcp://localhost:10002' --num-cluster '8,9,10'  --pcl-r 4 --seed 0 --warmup-epoch 50 --epochs 300


# 需要做的
1. 看懂相似度计算方式
2. 修改queue2的格式

# 相似度计算方式
1. 两两计算

# prompt
你是计算机视觉方面的专家，在对比学习上有深刻造诣。
我在修改对比学习的训练代码，在对比学习中，infonce是很常见的对比损失函数，对经过mlp之后的特征向量进行相似度计算，以moco为例，往往使用的方法是：
`l_global_pos = torch.einsum('nc,nc->n', [q_global, k_global]).unsqueeze(-1)  # positive samples`
`l_global_neg = torch.einsum('nc,ck->nk', [q_global, self.queue.clone().detach()])  # negative samples`
为了在dense层面执行对比学习，我将mlp拓展到了conv，对特征图进行相似度计算。
众所周知，在高维上进行余弦相似度计算不稳定，容易出现很多0值，一种常见的方法是flatten。然而，我不想丢失空间信息，所以我在探索新的相似度计算方法。
目前我考虑的是emd损失，来衡量两个分布之间的距离。
我遇到了一些问题：
1. 在dense层面，如何计算两个特征图之间的emd距离？这里比如x1=torch.randn(1,1024,7,7),x2=torch.randn(1,1024,7,7)
2. 在dense层面，在计算emd距离之后，l_dense_pos和l_dense_neg应该如何计算？据我所知，应该不能使用einsum了。

# 二轮
上述生成的代码使用scipy计算emd距离，和下面的torch计算结果差距有些大。是什么原因呢？我应该使用什么呢？
```
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
```