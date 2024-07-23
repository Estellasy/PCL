import torch
import torch.nn as nn
from random import sample

# 这里r需要修改 每次选择的原型数量和队列大小相同 应该为<原型数量 刚开始时没有那么多原型 queue_size要选择的原型数量多
class DenseCL(nn.Module):
    """
    Build a DenseCL model, change the MLP layer into dense layer.
    """
    def __init__(self, 
                 base_encoder,  # 也就是backbone
                 head=None, # head
                 dim=128, 
                 r=16384, 
                 m=0.999, 
                 T=0.1, 
                 loss_lambda=0.5,   # 损失权重
                 mlp=False):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(DenseCL, self).__init__()
        
        self.r = r
        self.m = m
        self.T = T
        
        # 创建编码器
        # 其中num_classes=dim是fc层的输出维度
        self.encoder_q = nn.Sequential(
            base_encoder(num_classes=dim),
            nn.Sequential()
        )
        
        self.encoder_k = nn.Sequential(
            base_encoder(num_classes=dim),
            nn.Sequential()
        )
        
        # 硬编码mlp层
        if mlp:
            dim_mlp = self.encoder_q[0].fc.weight.shape[1]
            # 删除原avgpool/fc层，并替换mlp
            self.encoder_q[0].avgpool = nn.Identity()
            self.encoder_q[0].fc = nn.Identity()
            self.encoder_k[0].avgpool = nn.Identity()
            self.encoder_k[0].fc = nn.Identity()
            
            # 更新neck
            self.encoder_q[1] = DenseNeck(dim_mlp, dim_mlp, dim)
            self.encoder_k[1] = DenseNeck(dim_mlp, dim_mlp, dim)
                
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # 初始化encoder_k的参数为encoder_q的参数
            param_k.requires_grad = False  # encoder_k不进行梯度更新
            
        # 创建两个队列 分别为global和dense
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.register_buffer("queue2",  torch.randn(dim, r))
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
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        if is_eval:
            # 获取encoder_k输出
            k_b = self.encoderk_features(im_q)
            # mlp层输出
            k, k_grid, _ = self.encoder_k[1](k_b)  # keys: NxC; NxCxS^2
            k = nn.functional.normalize(k, dim=1)   # global
            k_grid = nn.functional.normalize(k_grid, dim=1)  # dense
            return k, k_grid

        # 转为内存中的连续存储格式，提高访问效率
        im_q = im_q.contiguous()
        im_k = im_k.contiguous()
        # compute query features
        q_b = self.encoderq_features(im_q)  # backbone features
        q, q_grid, q2 = self.encoder_q[1](q_b)  # queries: NxC; NxCxS^2
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)
        
        q = nn.functional.normalize(q, dim=1)   # global
        q2 = nn.functional.normalize(q2, dim=1) # dense
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            
            # 获取encoder_k输出
            k_b = self.encoderk_features(im_k)  # encoder_k features
            # mlp层输出
            k, k_grid, k2 = self.encoder_k[1](k_b)  # keys: NxC; NxCxS^2
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)
            
            k = nn.functional.normalize(k, dim=1)   # global
            k2 = nn.functional.normalize(k2, dim=1)  # dense
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k_b = nn.functional.normalize(k_b, dim=1)
            
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)
            k_grid = self._batch_unshuffle_ddp(k_grid, idx_unshuffle)
            k_b = self._batch_unshuffle_ddp(k_b, idx_unshuffle)
            
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # 计算正样本对数似然 点积
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # 正样本
        # negative logits: NxK 矩阵乘法
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) # 负样本

        # feat point to set sim
        # 计算 q_b 和 k_b 之间的相似度矩阵
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)
        # 得到最大的索引
        densecl_sim_q = backbone_sim_matrix.max(dim=2)[1]   # NxS^2
        
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)  # NS^2x1
        
        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid,
                                            self.queue2.clone().detach()])
        
        # 损失计算
        logits_global = torch.cat([l_pos, l_neg], dim=1)  # Nx(1+K)
        logits_dense = torch.cat([l_pos_dense, l_neg_dense], dim=1)
        # apply temperature
        logits_global /= self.T
        logits_dense /= self.T
        # labels: postive key indicators
        # 每个样本的标签为 0 表示正样本
        labels_global = torch.zeros(logits_global.shape[0], dtype=torch.long).cuda()
        labels_dense = torch.zeros(logits_dense.shape[0], dtype=torch.long).cuda()
        
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue2(k2)
    
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
                all_proto_id_global = [i for i in range(im2cluster.max()+1)]
                neg_proto_id_global = set(all_proto_id_global) - set(pos_proto_id_global.tolist())
                
                # 随机采样r个原型
                neg_proto_id_global = sample(neg_proto_id_global, self.r)
                neg_prototypes_global = prototypes[neg_proto_id_global]
                
                proto_selected_global = torch.cat([pos_prototypes_global, neg_prototypes_global], dim=0)
                
                # compute prototypical logits
                logits_proto_global = torch.mm(q, proto_selected_global.t())
                
                # targets for prototype assignment
                labels_proto_global = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
                
                # scaling temperatures for the selected prototypes
                temp_proto_global = density[torch.cat([pos_proto_id_global,torch.LongTensor(neg_proto_id_global).cuda()],dim=0)]  
                logits_proto_global /= temp_proto_global
                
                proto_labels_global.append(labels_proto_global)
                proto_logits_global.append(logits_proto_global)
                
            result['global'] = [logits_global, labels_global, proto_logits_global, proto_labels_global]
        else:
            result['global'] = [logits_global, labels_global, None, None]

        if cluster_dense is not None:
            # TODO 这里采样r个原型修改逻辑 采样r个原型 这里r比较小
            # batch的淘汰机制需要确认
            proto_labels_dense = []
            proto_logits_dense = []
            for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_dense['im2cluster'], cluster_dense['centroids'], cluster_dense['density'])):
                # get positive prototypes
                pos_proto_id_dense = im2cluster[index]
                pos_prototypes_dense = prototypes[pos_proto_id_dense]
                
                # 采样负样本
                all_proto_id_dense = [i for i in range(im2cluster.max()+1)]
                neg_proto_id_dense = set(all_proto_id_dense) - set(pos_proto_id_dense.tolist())
                
                # 随机采样r个原型
                neg_proto_id_dense = sample(neg_proto_id_dense, self.r)
                neg_prototypes_dense = prototypes[neg_proto_id_dense]
                
                proto_selected_dense = torch.cat([pos_prototypes_dense, neg_prototypes_dense], dim=0)
                
                # compute prototypical logits
                logits_proto_dense = torch.mm(q, proto_selected_dense.t())
                
                # targets for prototype assignment
                labels_proto_dense = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
                
                # scaling temperatures for the selected prototypes
                temp_proto_dense = density[torch.cat([pos_proto_id_dense,torch.LongTensor(neg_proto_id_dense).cuda()],dim=0)]
                logits_proto_dense /= temp_proto_dense
                
                proto_labels_dense.append(labels_proto_dense)
                proto_logits_dense.append(logits_proto_dense)
            
            result['dense'] = [logits_dense, labels_dense, proto_logits_dense, proto_labels_dense]
        else:
            result['dense'] = [logits_dense, labels_dense, None, None]

        return result

            
    def encoderq_features(self, im_q):
        features = im_q
        # 提取avgpool层之前的特征
        for name, layer in self.encoder_q[0].named_children():
            if name == 'avgpool':
                break
            features = layer(features)
        return features
        
    def encoderk_features(self, im_k):
        features = im_k
        # 提取avgpool层之前的特征
        for name, layer in self.encoder_k[0].named_children():
            if name == 'avgpool':
                break
            features = layer(features)
        return features
    
    
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


class DenseNeck(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(DenseNeck, self).__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # mlp1 fc-relu-fc
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))
        # mlp2 conv1x1-relu-conv1x1
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1)
        )
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # avgpool
        avgpooled_x = self.avgpool(x)
        # mlp1
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))
        
        # mlp2
        x = self.mlp2(x) # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1) # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
        # 返回三个值 分别为原始mlp head/经过dense head后的mlp head/经过avgpool head后的mlp head
        return [avgpooled_x, x, avgpooled_x2]