- [x] 损失函数
- [x] 数据增强
- [ ] DetectionCL代码
- [ ] 训练主逻辑

# 计算 Wasserstein 距离

## 什么是最优传输问题
**移动概率质量函数**
把离散的概率分布想象成空间中分散的点的质量。我们可以观测这些带质量的点从一个分布移动到另一个分布需要做多少功。
![alt text](image.png)

定义一个度量标准，衡量移动所有点需要做的功。
=>引入耦合矩阵P（Coupling Matrix），表示从p(x)支撑集中的一个点到q(x)支撑集的一个点需要分配多少概率质量。对于均匀分布，规定每个点都有1/4的概率质量。

计算质量分配的过程需要做多少功=>引入距离矩阵
=>距离可以采用欧几里得距离

总成本计算
=>计算把点从一个支撑集移动到另一个支撑集中，使得成本较小的分配方式。

最优传输问题
=>解是所有耦合矩阵上的最低成本


## 生成数据的prompt
为了充分利用有限的训练数据并提高模型在表面缺陷检测任务中的表现，使用一种动态混合数据增强策略，结合CutPaste增强和MixUp增强。
请帮我写一个深度学习的数据加载工具，要求如下：
1. 数据的调用方式实例
"""
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

train_dataset = RandomImageFolderInstance(
        traindir,
        TwoCropsTransform(transforms.Compose(augmentation)))
"""
其中，RandomImageFolderInstance为你需要写的方法，TwoCropsTransform为数据增强方式，已写好。

2. 你需要做的：写RandomImageFolderInstance方法，
   假设训练数据集合为X={x1,x2,..,xn}，数量为n，batch大小为m。在每一批次的训练中，数据样本的构建方式如下：
    对于样本x_i \in x，从剩下的个样本中随机选择两个样本x_a和x_b，应用CutPaste增强和MixUp增强。
    a. CutPaste增强：对于x_a，应用CutPaste增强，首先利用二值分割快速定位前景和背景区域，随后在x_a的前景区域中随机cut一小块区域，paste到x_i的前景区域。
    b. MixUp增强: 对于x_b，应用MixUp增强，在通道维度对x_i和x_b进行线性变换混合。
    c. 混合CutPaste和MixUp：首先应用CutPaste增强，然后将结果与另一张图片进行MixUp增强。公式如下：
    MixUp(CutPaste(x_i, x_a), x_b) = \lambda CutPaste(x_i, x_a) + (1-\lambda)x_b

    RandomImageFolderInstance和下面的ImageFolderInstance输入输出保持一致：
    """
    class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index
    """


# 改写训练的prompt
TODO...

class Yolov8Head(nn.Module):
    def __init__(self, channels):
        super(Yolov8Head, self).__init__()
        # channels is a list of input channels for each stage
        _, c2, c3, c4 = channels
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_1 = C2f(c3 + c4, 512)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_2 = C2f(512 + c2, 256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.c2f_3 = C2f(256 + c2, 512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.c2f_4 = C2f(512 + c4, 1024)

    def forward(self, input_list):
        assert len(input_list) == 4
        _, x3, x2, x1 = input_list
        x = self.up1(x1)  # torch.Size([1, 2048, 40, 40])
        x = torch.cat((x, x2), dim=1)  # torch.Size([1, 3072, 40, 40])
        x = self.c2f_1(x)  # torch.Size([1, 512, 40, 40])
        hidden_x = x

        x = self.up2(x)  # torch.Size([1, 512, 80, 80])
        x = torch.cat((x, x3), dim=1)  # torch.Size([1, 1024, 80, 80])
        x = self.c2f_2(x)  # torch.Size([1, 256, 80, 80])
        p1 = x

        x = self.conv2(x)  # torch.Size([1, 256, 40, 40])
        x = torch.cat((x, hidden_x), dim=1)  # torch.Size([1, 768, 40, 40])
        x = self.c2f_3(x)  # torch.Size([1, 512, 40, 40])
        p2 = x

        x = self.conv3(x)  # torch.Size([1, 512, 20, 20])
        x = torch.cat((x, x1), dim=1)  # torch.Size([1, 2560, 20, 20])
        x = self.c2f_4(x)
        p3 = x  # torch.Size([1, 1024, 20, 20])

        return [p1, p2, p3]


这是我写的一个对比学习head，用于减小和后续下游任务的gap，但是现在由于输出是feature map list，需不方便后续操作，你需要帮我设计一种融合方式，融合p1/p2/p3，输出融合后的特征图

# dense训练的prompt
"""python
    def forward(self, im_q, im_k=None, is_eval=False, cluster_global=None, cluster_dense=None, index=None):
        """
        Args:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_global: global cluster assignments, centroids, and density
            cluster_dense: dense cluster assignments, centroids, and density
            index: indices for training samples

        Returns:
            logits, targets, proto_logits, proto_targets
        """
        if is_eval:
            # 获取encoder_k输出
            k_b_list = self.encoderk_features(im_q)
            # mlp层输出
            k_global = self.encoder_k[1](k_b_list[-1])
            # yolo head输出
            k_dense = self.encoder_k[2](k_b_list)
            k_global = nn.functional.normalize(k_global, dim=1)
            k_dense = nn.functional.normalize(k_dense, dim=1)
            return k_global, k_dense

        # 转为内存中的连续存储格式，提高访问效率
        im_q = im_q.contiguous()
        im_k = im_k.contiguous()
        # compute query features
        q_b_list = self.encoderq_features(im_q)  # backbone features
        q_global = self.encoder_q[1](q_b_list[-1])
        q_dense = self.encoder_q[2](q_b_list)

        q_global = nn.functional.normalize(q_global, dim=1)
        q_dense = nn.functional.normalize(q_dense, dim=1)

        # compute key features
        with torch.no_grad():   # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            # 获取encoder_k输出
            k_b_list = self.get_encoderk_features(im_k)
            # mlp输出
            k_global = self.encoder_k[1](k_b_list[-1])
            k_dense = self.encoder_k[2](k_b_list)

            k_global = nn.functional.normalize(k_global, dim=1)  # global
            k_dense = nn.functional.normalize(k_dense, dim=1)  # dense

            # undo shuffle
            k_global = self._batch_unshuffle_ddp(k_global, idx_unshuffle)
            k_dense = self._batch_unshuffle_ddp(k_dense, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # 计算正样本对数似然 点积
        l_global_pos = torch.einsum('nc,nc->n', [q_global, k_global]).unsqueeze(-1)  # 正样本
        # negative logits: NxK矩阵惩罚
        l_global_neg = torch.einsum('nc,ck->nk', [q_global, self.queue.clone().detach()]) # 负样本

        l_dense_pos = torch.einsum()
        l_dense_neg = torch.einsum()

        # 损失计算
        logits_global = torch.cat([l_global_pos, l_global_neg], dim=1)  # Nx(1+K)
        logits_dense = torch.cat([l_dense_pos, l_dense_neg], dim=1)
        # apply temperature
        logits_global /= self.T
        logits_dense /= self.T
        # labels: postive key indicators
        # 每个样本的标签为 0 表示正样本
        labels_global = torch.zeros(logits_global.shape[0], dtype=torch.long).cuda()
        labels_dense = torch.zeros(logits_dense.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k_global)
        self._dequeue_and_enqueue2(k_dense)

        result = dict()

        # prototypical contrast
        if cluster_global is not None:
            proto_labels_global = []
            proto_logits_global = []
            for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_global['im2cluster'], cluster_global['centroids'], cluster_global['density'])):
                # get positive prototypes
                pos_proto_id_global = im2cluster[index]
                pos_prototypes_global = prototypes[pos_proto_id_global]

                # 采样负样本
                all_proto_id_global = [i for i in range(im2cluster.max() + 1)]
                neg_proto_id_global = set(all_proto_id_global) - set(pos_proto_id_global.tolist())

                # 随机采样r个原型
                neg_proto_id_global = sample(neg_proto_id_global, self.r)
                neg_prototypes_global = prototypes[neg_proto_id_global]

                proto_selected_global = torch.cat([pos_prototypes_global, neg_prototypes_global], dim=0)

                # compute prototypical logits
                logits_proto_global = torch.mm(q_global, proto_selected_global.t())

                # targets for prototype assignment
                labels_proto_global = torch.linspace(0, q_global.size(0) - 1, steps=q_global.size(0)).long().cuda()

                # scaling temperatures for the selected prototypes
                temp_proto_global = density[torch.cat([pos_proto_id_global, torch.LongTensor(neg_proto_id_global).cuda()], dim=0)]
                logits_proto_global /= temp_proto_global

                proto_labels_global.append(labels_proto_global)
                proto_logits_global.append(logits_proto_global)

            result['global'] = [logits_global, labels_global, proto_logits_global, proto_labels_global]
        else:
            result['global'] = [logits_global, labels_global, None, None]


        if cluster_dense is not None:
            pass
"""
这是原型对比学习的部分训练代码，我将global级别的原型对比学习拓展到dense级别，dense特征图由yolohead得到，大小为torch.Size([1, 1024, 20, 20])。
你需要帮我补充代码，具体来说，有两个部分：
1. """
# compute logits
    # Einstein sum is more intuitive
    # positive logits: Nx1
    # 计算正样本对数似然 点积
    l_global_pos = torch.einsum('nc,nc->n', [q_global, k_global]).unsqueeze(-1)  # 正样本
    # negative logits: NxK矩阵惩罚
    l_global_neg = torch.einsum('nc,ck->nk', [q_global, self.queue.clone().detach()]) # 负样本

    l_dense_pos = torch.einsum()
    l_dense_neg = torch.einsum()
"""
拓展l_dense。

2. """
if cluster_dense is not None:
        pass"""
补充cluster_dense的聚类方式

# 修改训练逻辑的prompt
下面是数据的训练逻辑，给出了训练流程，以及global feature的处理，请帮我补充dense级别的处理。
主要有三个地方：
1. """
   # dense
            cluster_result_dense = {'im2cluster':[],'centroids':[],'density':[]}
   """

2. """
if args.gpu == 0:
                global_features[torch.norm(global_features,dim=1)>1.5] /= 2 # account for the few samples that are computed twice  
                global_features = global_features.numpy()
                cluster_result_global = run_kmeans(global_features, args)

                # dense
"""

3. """
def compute_features(eval_loader, model, args):
    print('Computing features')
    model.eval()
    # global features 
    global_features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda()
    # dense features
    dense_features = 

    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat_global, feat_dense = model(images, eval=True)
            global_features[index] = feat_global
            # local

    dist.barrier()
    dist.all_reduce(global_features, op=dist.ReduceOp.SUM)   

    return global_features.cpu(), dense_features.cpu()
"""

每个epoch的cluster代码如下
"""python
for epoch in range(args.start_epoch, args.epochs):
        cluster_result_global = None
        cluster_result_dense = None

        if epoch >= args.warmup_epoch:
            # compute momentum features for center-cropped images
            global_features, dense_features = compute_features(eval_loader, model, args)

            # placeholder for clustering result
            cluster_result_global = {'im2cluster':[],'centroids':[],'density':[]}
            for num_cluster in args.num_cluster:
                cluster_result_global['im2cluster'].append(torch.zeros(len(eval_dataset),dtype=torch.long).cuda())
                cluster_result_global['centroids'].append(torch.zeros(int(num_cluster),args.low_dim).cuda())
                cluster_result_global['density'].append(torch.zeros(int(num_cluster)).cuda()) 

            # dense
            cluster_result_dense = {'im2cluster':[],'centroids':[],'density':[]}


            if args.gpu == 0:
                global_features[torch.norm(global_features,dim=1)>1.5] /= 2 # account for the few samples that are computed twice  
                global_features = global_features.numpy()
                cluster_result_global = run_kmeans(global_features, args)

                # dense
            
            dist.barrier()  
            # broadcast clustering result
            for k, data_list in cluster_result_global.items():
                for data_tensor in data_list:                
                    dist.broadcast(data_tensor, 0, async_op=False)     

            for k, data_list in cluster_result_dense.items():
                for data_tensor in data_list:                
                    dist.broadcast(data_tensor, 0, async_op=False)     
"""



这是main_detection的完整代码，
```python
"""
@Description : DetectionCL
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/21 15:33:48
"""

import torch
import torch.nn as nn
from random import sample
import sys
sys.path.append("/home/siyi/project/PCL/pcl")
from head import Yolov8Head, MlpHead


class DetectionCL(nn.Module):
    """
    Build a DetectionCL model, change the MLP layer into detection layer.
    """

    def __init__(self, base_encoder, dim=128, r=10, m=0.999, T=0.1, loss_lambda=0.5, mlp=True) -> None:
        super(DetectionCL, self).__init__()

        self.r, self.m, self.T = r, m, T

        # 创建编码器 其中num_classes=dim是fc层的输出维度
        self.encoder_q = nn.Sequential(
            base_encoder(num_classes=dim),
            nn.Sequential(),    # mlp
            nn.Sequential()     # detection head
        )
        self.encoder_k = nn.Sequential(
            base_encoder(num_classes=dim),
            nn.Sequential(),    # mlp
            nn.Sequential()     # detection head
        )

        # 硬编码mlp层
        if mlp:
            dim_mlp = self.encoder_q[0].fc.weight.shape[1]
            # 删除原avgpool/fc层 并替换mlp
            self.encoder_q[0].avgpool = nn.Identity()
            self.encoder_q[0].avgpool = nn.Identity()
            self.encoder_q[0].fc = nn.Identity()
            self.encoder_k[0].avgpool = nn.Identity()
            self.encoder_k[0].fc = nn.Identity()

            # mlp层
            self.encoder_k[1] = MlpHead(dim_mlp, dim_mlp, dim)
            self.encoder_q[1] = MlpHead(dim_mlp, dim_mlp, dim)

            # head层
            channels = [256, 512, 1024, 2048]
            self.encoder_q[2] = Yolov8Head(channels)
            self.encoder_k[2] = Yolov8Head(channels)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # 初始化encoder_k的参数为encoder_q的参数
            param_k.requires_grad = False  # encoder_k不进行梯度更新

        # 创建两个队列 分别为global和dense
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue2", torch.randn(dim, r))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k=None, is_eval=False, cluster_global=None, cluster_dense=None, index=None):
        """
        Args:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_global: global cluster assignments, centroids, and density
            cluster_dense: dense cluster assignments, centroids, and density
            index: indices for training samples

        Returns:
            logits, targets, proto_logits, proto_targets
        """
        if is_eval:
            # 获取encoder_k输出
            k_b_list = self.get_encoderk_features(im_q)
            print("k_b_list[-1]:", k_b_list[-1].shape)  # torch.Size([20, 2048, 7, 7])
            # mlp层输出
            k_global = self.encoder_k[1](k_b_list[-1])
            print("k_global:", k_global.shape)
            # yolo head输出
            k_dense = self.encoder_k[2](k_b_list)
            k_global = nn.functional.normalize(k_global, dim=1)
            k_dense = nn.functional.normalize(k_dense, dim=1)
            print(k_global.shape, k_dense.shape)
            return k_global, k_dense

        # 转为内存中的连续存储格式，提高访问效率
        im_q = im_q.contiguous()
        im_k = im_k.contiguous()
        # compute query features
        q_b_list = self.get_encoderq_features(im_q)  # backbone features
        q_global = self.encoder_q[1](q_b_list[-1])
        q_dense = self.encoder_q[2](q_b_list)

        q_global = nn.functional.normalize(q_global, dim=1)
        q_dense = nn.functional.normalize(q_dense, dim=1)

        # compute key features
        with torch.no_grad():   # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            # 获取encoder_k输出
            k_b_list = self.get_encoderk_features(im_k)
            # mlp输出
            k_global = self.encoder_k[1](k_b_list[-1])
            k_dense = self.encoder_k[2](k_b_list)

            k_global = nn.functional.normalize(k_global, dim=1)  # global
            k_dense = nn.functional.normalize(k_dense, dim=1)  # dense

            # undo shuffle
            k_global = self._batch_unshuffle_ddp(k_global, idx_unshuffle)
            k_dense = self._batch_unshuffle_ddp(k_dense, idx_unshuffle)

        # Compute logits
        # Positive logits: Nx1 for global and dense features
        l_global_pos = torch.einsum('nc,nc->n', [q_global, k_global]).unsqueeze(-1)  # positive samples
        l_global_neg = torch.einsum('nc,ck->nk', [q_global, self.queue.clone().detach()])  # negative samples

        print("l_global_pos:", l_global_pos.shape)  # torch.Size([20, 1])
        print("l_global_neg:", l_global_neg.shape)

        # For dense features, we need to reshape and compute logits accordingly
        # Reshape dense features to (N, C, H*W)
        q_dense_flat = q_dense.view(q_dense.size(0), q_dense.size(1), -1).permute(0, 2, 1)  # (N, H*W, C)
        k_dense_flat = k_dense.view(k_dense.size(0), k_dense.size(1), -1).permute(0, 2, 1)  # (N, H*W, C)


        print("q_dense_flat:", q_dense_flat.shape)  # torch.Size([4, 49, 1024])
        print("q_dense:", q_dense.shape)
        # Positive logits for dense features: Nx(H*W)   
        l_dense_pos = torch.einsum('nqc,nqc->nq', [q_dense_flat, k_dense_flat]).view(q_dense.size(0), -1)
        # Negative logits for dense features: Nx(H*W) x (K)
        l_dense_neg = torch.einsum('nqc,ck->nqk', [q_dense_flat, self.queue.clone().detach()]).view(q_dense.size(0), -1,
                                                                                                    self.queue.size(0))
        print("l_dense_pos:", l_dense_pos.shape)  # torch.Size([20, 1])

        logits_global = torch.cat([l_global_pos, l_global_neg], dim=1)  # Nx(1+K)
        logits_dense = torch.cat([l_dense_pos, l_dense_neg], dim=1)  # Nx(H*W+H*W*K)
        # apply temperature
        logits_global /= self.T
        logits_dense /= self.T
        # labels: postive key indicators
        # 每个样本的标签为 0 表示正样本
        labels_global = torch.zeros(logits_global.shape[0], dtype=torch.long).cuda()
        labels_dense = torch.zeros(logits_dense.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k_global)
        self._dequeue_and_enqueue2(k_dense)

        result = dict()

        # prototypical contrast
        if cluster_global is not None:
            proto_labels_global = []
            proto_logits_global = []
            for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_global['im2cluster'], cluster_global['centroids'], cluster_global['density'])):
                # get positive prototypes
                pos_proto_id_global = im2cluster[index]
                pos_prototypes_global = prototypes[pos_proto_id_global]

                # 采样负样本
                all_proto_id_global = [i for i in range(im2cluster.max() + 1)]
                neg_proto_id_global = set(all_proto_id_global) - set(pos_proto_id_global.tolist())

                # 随机采样r个原型
                neg_proto_id_global = sample(neg_proto_id_global, self.r)
                neg_prototypes_global = prototypes[neg_proto_id_global]

                proto_selected_global = torch.cat([pos_prototypes_global, neg_prototypes_global], dim=0)

                # compute prototypical logits
                logits_proto_global = torch.mm(q_global, proto_selected_global.t())

                # targets for prototype assignment
                labels_proto_global = torch.linspace(0, q_global.size(0) - 1, steps=q_global.size(0)).long().cuda()

                # scaling temperatures for the selected prototypes
                temp_proto_global = density[torch.cat([pos_proto_id_global, torch.LongTensor(neg_proto_id_global).cuda()], dim=0)]
                logits_proto_global /= temp_proto_global

                proto_labels_global.append(labels_proto_global)
                proto_logits_global.append(logits_proto_global)

            result['global'] = [logits_global, labels_global, proto_logits_global, proto_labels_global]
        else:
            result['global'] = [logits_global, labels_global, None, None]


        if cluster_dense is not None:
            proto_labels_dense = []
            proto_logits_dense = []
            for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_dense['im2cluster'], cluster_dense['centroids'], cluster_dense['density'])):
                pos_proto_id_dense = im2cluster[index]
                pos_prototypes_dense = prototypes[pos_proto_id_dense]

                all_proto_id_dense = [i for i in range(im2cluster.max() + 1)]
                neg_proto_id_dense = set(all_proto_id_dense) - set(pos_proto_id_dense.tolist())
                neg_proto_id_dense = sample(neg_proto_id_dense, self.r)
                neg_prototypes_dense = prototypes[neg_proto_id_dense]

                proto_selected_dense = torch.cat([pos_prototypes_dense, neg_prototypes_dense], dim=0)
                logits_proto_dense = torch.mm(q_dense_flat.view(-1, q_dense_flat.size(2)),
                                              proto_selected_dense.t()).view(q_dense_flat.size(0), -1,
                                                                             proto_selected_dense.size(0))

                labels_proto_dense = torch.linspace(0, q_dense_flat.size(0) - 1,
                                                    steps=q_dense_flat.size(0)).long().cuda()
                temp_proto_dense = density[
                    torch.cat([pos_proto_id_dense, torch.LongTensor(neg_proto_id_dense).cuda()], dim=0)]
                logits_proto_dense /= temp_proto_dense

                proto_labels_dense.append(labels_proto_dense)
                proto_logits_dense.append(logits_proto_dense)

            result['dense'] = [logits_dense, labels_dense, proto_logits_dense, proto_labels_dense]
        else:
            result['dense'] = [logits_dense, labels_dense, None, None]

        return result

    def get_encoderq_features(self, im_q):
        features = im_q
        output_list = []
        for name, layer in self.encoder_q[0].named_children():
            if name == 'avgpool':
                break
            features = layer(features)
            if name.startswith("layer"):
                output_list.append(features)
        return output_list

    def get_encoderk_features(self, im_k):
        features = im_k
        output_list = []
        for name, layer in self.encoder_k[0].named_children():
            if name == 'avgpool':
                break
            features = layer(features)
            if name.startswith("layer"):
                output_list.append(features)
        return output_list

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
```

我正在训练main_detection，代码在这里出现了问题，报错如下：
"""
-- Process 0 terminated with the following error:
Traceback (most recent call last):
  File "/home/siyi/miniconda3/envs/detection/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap
    fn(i, *args)
  File "/home/siyi/project/PCL/main_detection.py", line 353, in main_worker
    train(train_loader, model, criterion, emd_loss_fn, optimizer, epoch, args, cluster_result_global, cluster_result_dense)
  File "/home/siyi/project/PCL/main_detection.py", line 391, in train
    result = model(im_q=images[0], im_k=images[1], cluster_global=cluster_result_global, cluster_dense=cluster_result_dense, index=index)
  File "/home/siyi/miniconda3/envs/detection/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1190, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/siyi/miniconda3/envs/detection/lib/python3.8/site-packages/torch/nn/parallel/distributed.py", line 1040, in forward
    output = self._run_ddp_forward(*inputs, **kwargs)
  File "/home/siyi/miniconda3/envs/detection/lib/python3.8/site-packages/torch/nn/parallel/distributed.py", line 1000, in _run_ddp_forward
    return module_to_run(*inputs[0], **kwargs[0])
  File "/home/siyi/miniconda3/envs/detection/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1190, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/siyi/project/PCL/pcl/detection_builder.py", line 233, in forward
    l_dense_neg = torch.einsum('nqc,ck->nqk', [q_dense_flat, self.queue.clone().detach()]).view(q_dense.size(0), -1,
  File "/home/siyi/miniconda3/envs/detection/lib/python3.8/site-packages/torch/functional.py", line 373, in einsum
    return einsum(equation, *_operands)
  File "/home/siyi/miniconda3/envs/detection/lib/python3.8/site-packages/torch/functional.py", line 378, in einsum
    return _VF.einsum(equation, operands)  # type: ignore[attr-defined]
RuntimeError: einsum(): subscript c has size 128 for operand 1 which does not broadcast with previously seen size 1024
"""
请帮我检查一下问题是什么，并帮我修订


你的意思是，代码问题出现在这里：
l_dense_neg = torch.einsum('nqc,ck->nqk', [q_dense_flat, self.queue.clone().detach()]).view(q_dense.size(0), -1,

因为dense和global的大小不同，但是存储在同一个queue中。

这样看来，你应该帮我修正queue部分的代码。queue部分的代码如下：

创建两个队列 分别为global和dense
python
    self.register_buffer("queue", torch.randn(dim, r))
    self.queue = nn.functional.normalize(self.queue, dim=0)
    self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    self.register_buffer("queue2", torch.randn(dim, r))
    self.queue2 = nn.functional.normalize(self.queue2, dim=0)
    self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))
看上去dense应该存储在queue2中，但是queue2的大小有问题，请帮我修正。
print("q_dense_flat:", q_dense_flat.shape) # torch.Size([4, 49, 1024])


# debug prompt
代码在下面的地方报错：
"""
"""
报错信息如下