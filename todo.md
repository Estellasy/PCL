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

