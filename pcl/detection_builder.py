"""
@Description : DetectionCL
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/21 15:33:48
"""

import torch
import torch.nn as nn
from random import sample
import sys

sys.path.append("../PCL/pcl")
from head import Yolov8Head, MlpHead
from backbone import resnet50
from loss import SinkhornSim


class DetectionCL(nn.Module):
    """
    Build a DetectionCL model, change the MLP layer into detection layer.
    """

    def __init__(self, base_encoder, dim=128, r=10, m=0.999, T=0.1, loss_lambda=0.5, mlp=True) -> None:
        super(DetectionCL, self).__init__()

        self.r, self.m, self.T = r, m, T

        # 创建编码器 其中num_classes=dim是fc层的输出维度
        self.encoder_q = nn.Sequential(
            resnet50(num_classes=dim),
            nn.Sequential(),  # mlp
            nn.Sequential()  # detection head
        )
        self.encoder_k = nn.Sequential(
            resnet50(num_classes=dim),
            nn.Sequential(),  # mlp
            nn.Sequential()  # detection head
        )

        # 硬编码mlp层
        if mlp:
            dim_mlp = 2048

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

        channels = 1024  # 这是 q_dense_flat 的通道维度
        height, width = 7, 7  # 这是 q_dense_flat 的空间维度
        self.register_buffer("queue2", torch.randn(channels, self.r, height, width))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        print("queue2:", self.queue2.shape)
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
        assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue2.size(1) % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size, :, :] = keys.permute(1, 0, 2, 3)
        ptr = (ptr + batch_size) % self.queue2.size(1)  # move pointer

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
            # print("k_b_list[-1]:", k_b_list[-1].shape)  # torch.Size([20, 2048, 7, 7])
            # mlp层输出
            k_global = self.encoder_k[1](k_b_list[-1])
            # print("k_global:", k_global.shape)
            # yolo head输出
            k_dense = self.encoder_k[2](k_b_list)
            k_global = nn.functional.normalize(k_global, dim=1)
            k_dense = nn.functional.normalize(k_dense, dim=1)
            # print(k_global.shape, k_dense.shape)
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
        with torch.no_grad():  # no gradient to keys
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

        # print("l_global_pos:", l_global_pos.shape)  # torch.Size([20, 1])
        # print("l_global_neg:", l_global_neg.shape)

        # For dense features, we need to reshape and compute logits accordingly
        # Reshape dense features to (N, C, H*W)
        print("q_dense:", q_dense.shape)  # torch.Size([20, 2048, 7, 7])
        print("self.queue2.clone().detach():", self.queue2.clone().detach().shape)

        # 在这里修改l_dense_pos和l_dense_neg的计算方式
        l_dense_pos = []
        ssim = SinkhornSim()
        for i in range(q_dense.size(0)):  # (N, C, H, W)
            q_dense_i = q_dense[i].unsqueeze(0)  # (1, C, H, W)
            k_dense_i = k_dense[i].unsqueeze(0)  # (1, C, H, W)
            l_dense_pos.append(ssim(q_dense_i, k_dense_i))
        l_dense_pos = torch.stack(l_dense_pos, dim=0)  # (N, 1)

        l_dense_neg = []
        for neg_sample in self.queue2.clone().detach(): # 对于每个负样本
            neg_sample = neg_sample.unsqueeze(0)  # (1, C, H, W)
            temp_neg = []
            for i in range(q_dense.size(0)):  # (N, C, H, W)
                q_dense_i = q_dense[i].unsqueeze(0)  # (1, C, H, W)
                temp_neg.append(ssim(q_dense_i, neg_sample))
            l_dense_neg.append(torch.stack(temp_neg, dim=0))  # (N, 1)
        # 这里l_dense_neg的维度有待商榷
        l_dense_neg = torch.stack(l_dense_neg).permute(1, 0).squeeze(-1)  # (N, K)


        q_dense_flat = q_dense.view(q_dense.size(0), q_dense.size(1), -1).permute(0, 2, 1)  # (N, H*W, C)
        k_dense_flat = k_dense.view(k_dense.size(0), k_dense.size(1), -1).permute(0, 2, 1)  # (N, H*W, C)

        # print("q_dense_flat:", q_dense_flat.shape)  # torch.Size([4, 49, 1024])
        # print("q_dense:", q_dense.shape)    # torch.Size([4, 1024, 7, 7])
        # Positive logits for dense features: Nx(H*W)
        l_dense_pos = torch.einsum('nqc,nqc->nq', [q_dense_flat, k_dense_flat]).view(q_dense.size(0), -1)
        l_dense_pos = torch.unsqueeze(l_dense_pos, -1)
        l_dense_neg = torch.einsum('nqc,ckhw->nqk', [q_dense_flat, self.queue2.clone().detach()]).view(q_dense.size(0),
                                                                                                       -1,
                                                                                                       self.queue2.size(
                                                                                                           1))
        # Negative logits for dense features: Nx(H*W) x (K)
        # l_dense_neg = torch.einsum('nqc,ck->nqk', [q_dense_flat, self.queue2.clone().detach()])
        # l_dense_neg = l_dense_neg.view(q_dense.size(0), -1, self.queue2.size(1))
        # print("l_dense_pos:", l_dense_pos.shape)  # torch.Size([4, 49])
        # print("l_dense_neg:", l_dense_neg.shape)  # torch.Size([4, 49])

        logits_global = torch.cat([l_global_pos, l_global_neg], dim=1)  # Nx(1+K)
        logits_dense = torch.cat([l_dense_pos, l_dense_neg], dim=2)  # Nx(H*W+H*W*K)
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
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(cluster_global['im2cluster'], cluster_global['centroids'], cluster_global['density'])):
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
                temp_proto_global = density[
                    torch.cat([pos_proto_id_global, torch.LongTensor(neg_proto_id_global).cuda()], dim=0)]
                logits_proto_global /= temp_proto_global

                proto_labels_global.append(labels_proto_global)
                proto_logits_global.append(logits_proto_global)

            result['global'] = [logits_global, labels_global, proto_logits_global, proto_labels_global]
        else:
            result['global'] = [logits_global, labels_global, None, None]

        if cluster_dense is not None:
            proto_labels_dense = []
            proto_logits_dense = []
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(cluster_dense['im2cluster'], cluster_dense['centroids'], cluster_dense['density'])):
                pos_proto_id_dense = im2cluster[index]
                pos_prototypes_dense = prototypes[pos_proto_id_dense]

                # print(f"pos_proto_id_dense shape: {pos_proto_id_dense.shape}")
                # print(f"pos_prototypes_dense shape: {pos_prototypes_dense.shape}")

                all_proto_id_dense = [i for i in range(im2cluster.max() + 1)]
                neg_proto_id_dense = set(all_proto_id_dense) - set(pos_proto_id_dense.tolist())
                neg_proto_id_dense = sample(neg_proto_id_dense, self.r)
                neg_prototypes_dense = prototypes[neg_proto_id_dense]

                # # print(f"neg_proto_id_dense length: {len(neg_proto_id_dense)}")
                # # print(f"neg_prototypes_dense shape: {neg_prototypes_dense.shape}")

                proto_selected_dense = torch.cat([pos_prototypes_dense, neg_prototypes_dense], dim=0)
                # # print(f"proto_selected_dense shape: {proto_selected_dense.shape}")

                # print(q_dense_flat.shape)
                # temp = q_dense_flat.reshape(q_dense_flat.size(0), -1)
                # print(temp.shape)
                logits_proto_dense = torch.mm(q_dense_flat.reshape(q_dense_flat.size(0), -1),
                                              proto_selected_dense.t()).reshape(q_dense_flat.size(0), -1,
                                                                                proto_selected_dense.size(0))
                # print(f"logits_proto_dense shape: {logits_proto_dense.shape}")

                labels_proto_dense = torch.linspace(0, q_dense_flat.size(0) - 1,
                                                    steps=q_dense_flat.size(0)).long().cuda()
                # print(f"labels_proto_dense shape: {labels_proto_dense.shape}")

                temp_proto_dense = density[
                    torch.cat([pos_proto_id_dense, torch.LongTensor(neg_proto_id_dense).cuda()], dim=0)]
                # print(f"temp_proto_dense shape: {temp_proto_dense.shape}")

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
