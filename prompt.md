# args转成读取配置文件
请帮我写一段代码，将parse_args转成读取配置文件的形式，配置文件使用yaml格式。请帮我选择合适的库，并且配置。

源代码如下：
```python
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pcl.detection_builder
import pcl.detection_loader
from pcl.loss import EMDLoss

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--low-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--pcl-r', default=16384, type=int,
                    help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')

parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--num-cluster', default='25000,50000,100000', type=str, 
                    help='number of clusters')
parser.add_argument('--warmup-epoch', default=20, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                    help='experiment directory')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.num_cluster = args.num_cluster.split(',')

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
```

更新配置文件yaml如下：
{{ code here }} 

更新源代码如下:
{{ code here }}

# cluster dense
你是一个对比学习专家，对原型对比学习PCL颇有造诣。
我在做原型对比学习的实验，想要将instance级别的原型对比学习拓展到feature层级。
下面是instance层级的代码：
'''
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
'''

我想要拓展到dense级别，请帮我补充代码：
'''python
        if cluster_dense is not None:
            proto_labels_dense = []
            proto_logits_dense = []
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(cluster_dense['im2cluster'], cluster_dense['centroids'], cluster_dense['density'])):
                    {{ code here }}
'''


# 聚类
我发现，如果我在密集特征执行对比，我必须获得密集原型。但似乎使用faiss聚类只能对一维tensor进行kmeans聚类。
我现在想做的是，写一个对大小为(1024，7,7)特征进行聚类的函数，在相似度计算中，使用emd距离，就像前面提到的SinkhornSim()。但是我不知道要怎么做，不能直接调用faiss聚类。

我需要你的帮助，我需要你帮我完成上述聚类功能，实现对（1024，7，7）特征聚类。在相似度计算时，你可以调用SinkornSim(x1,x2)进行计算。请帮我补充run_kmeans_dense的代码。

下面是使用faiss进行global级别run_kmeans的代码，你可以参考。
"""python

def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': []}

    for seed, num_cluster in enumerate(args.num_cluster):
        # initialize faiss clutering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = args.temperature * density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results
"""

对于results_dense['centroids']，里面的元素shape应该是(1024,7,7)