"""
@Description :
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/21 15:33:48
"""

import torch
import torch.nn as nn
from random import sample
from head import Yolov8Head


class DetectionCL(nn.Module):
    """
    Build a DetectionCL model, change the MLP layer into detection layer.
    """
    def __init__(self, base_encoder, dim=128, r=10, m=0.999, T=0.1, loss_lambda=0.5,  mlp=True) -> None:
        super(DetectionCL, self).__init__()

        self.r, self.m, self.T = r, m, T

        # 创建编码器 其中num_classes=dim是fc层的输出维度
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
            # 删除原avgpool/fc层 并替换mlp
            self.encoder_q[0].avgpool = nn.Identity()
            self.encoder_q[0].avgpool = nn.Identity()
            self.encoder_q[0].fc = nn.Identity()
            self.encoder_k[0].avgpool = nn.Identity()
            self.encoder_k[0].fc = nn.Identity()

            # 更新head
            channels = [256, 512, 1024, 2048]
            self.encoder_q[1] = Yolov8Head(channels)
            self.encoder_k[1] = Yolov8Head(channels)


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
        pass


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