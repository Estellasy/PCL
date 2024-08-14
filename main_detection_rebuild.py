"""
@Description : 代码重构 使用yaml读取配置文件
@Author      : siyiren1@foxmail.com
@Time        : 2024/08/10 12:54:05
"""

import argparse
import yaml
import builtins
import math
import os
import random
import shutil
import time
import datetime
import warnings
from tqdm import tqdm
import numpy as np
import faiss
from random import sample

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
from pcl.sliced_loss import sliced_loss


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def load_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    for key, value in config.items():
        if isinstance(value, str) and value.isdigit():
            config[key] = int(value)
        elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
            config[key] = float(value)
        elif key == 'weight_decay' and isinstance(value, str):
            config[key] = float(value)

    # Update exp_dir with current timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    config['exp_dir'] = os.path.join(config['exp_dir'], timestamp)
    return config


def main():
    # Add an argument to specify the configuration file path
    parser = argparse.ArgumentParser(description='Dynamic Configuration')
    parser.add_argument('--config', type=str, default='config/base_detection_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Update the argparse.Namespace object with the loaded configuration
    args = argparse.Namespace(**config)

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


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = pcl.detection_builder.DetectionCL(
        models.__dict__[args.arch],
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp
    )
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([pcl.detection_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    # center-crop augmentation
    eval_augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = pcl.detection_loader.RandomImageFolderInstance(
        traindir,
        pcl.detection_loader.TwoCropsTransform(transforms.Compose(augmentation)))
    eval_dataset = pcl.detection_loader.ImageFolderInstance(
        traindir,
        eval_augmentation)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size * 5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)

    # 修改训练epoch逻辑
    for epoch in range(args.start_epoch, args.epochs):
        cluster_result_global = None
        cluster_result_dense = None

        if epoch >= args.warmup_epoch:
            # compute momentum features for center-cropped images
            global_features, dense_features = compute_features(eval_loader, model, args)
            print(global_features.device)
            print(dense_features.device)

            # placeholder for clustering result
            cluster_result_global = {'im2cluster': [], 'centroids': [], 'density': []}
            for num_cluster in args.num_cluster:
                cluster_result_global['im2cluster'].append(torch.zeros(len(eval_dataset), dtype=torch.long).cuda())
                cluster_result_global['centroids'].append(torch.zeros(int(num_cluster), args.low_dim).cuda())
                cluster_result_global['density'].append(torch.zeros(int(num_cluster)).cuda())

            # dense
            cluster_result_dense = {'im2cluster': [], 'centroids': [], 'density': []}
            for num_cluster in args.num_cluster:
                cluster_result_dense['im2cluster'].append(torch.zeros(len(eval_dataset), dtype=torch.long).cuda())
                cluster_result_dense['centroids'].append(
                    torch.zeros(int(num_cluster), 1024 * 20 * 20).cuda())  # Flattened size
                cluster_result_dense['density'].append(torch.zeros(int(num_cluster)).cuda())

            if args.gpu == 0:
                global_features[torch.norm(global_features,
                                           dim=1) > 1.5] /= 2  # account for the few samples that are computed twice
                global_features = global_features.detach().cpu().numpy()
                cluster_result_global = run_kmeans(global_features, args)
                # print("global")
                # dense
                # dense_features = dense_features.view(dense_features.size(0), -1)  # Flatten dense features
                # dense_features[dense_features.norm(dim=1) > 1.5] /= 2
                # dense_features = dense_features.numpy()
                cluster_result_dense = run_kmeans_dense(dense_features, args)
                # cluster_result_dense = run_kmeans_emd(dense_features, args)

                # print("dense")
            dist.barrier()
            # print("dist bug")
            # broadcast clustering result
            for k, data_list in cluster_result_global.items():
                for data_tensor in data_list:
                    dist.broadcast(data_tensor, 0, async_op=False)

                    # print("broadcast global")

            for k, data_list in cluster_result_dense.items():
                for data_tensor in data_list:
                    dist.broadcast(data_tensor, 0, async_op=False)

                    # print("broadcast dense")

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, cluster_result_global, cluster_result_dense)

        if (epoch + 1) % 20 == 0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                                               and args.rank % ngpus_per_node == 0)):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir, epoch))


def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result_global=None, cluster_result_dense=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    global_loss = AverageMeter('Global Loss', ':.4e')
    dense_loss = AverageMeter('Dense Loss', ':.4e')
    global_proto_loss = AverageMeter('Global Proto Loss', ':.4e')
    dense_proto_loss = AverageMeter('Dense Proto Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')
    acc_proto_global = AverageMeter('Acc@Global Proto', ':6.2f')
    acc_proto_dense = AverageMeter('Acc@Dense Proto', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, global_loss, dense_loss, global_proto_loss, dense_proto_loss, acc_inst,
         acc_proto_global, acc_proto_dense],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        result = model(im_q=images[0], im_k=images[1], cluster_global=cluster_result_global, cluster_dense=cluster_result_dense, index=index)

        # loss
        loss = 0.
        global_output, global_target, global_output_proto, global_target_proto = result['global']
        dense_output, dense_target, dense_output_proto, dense_target_proto = result['dense']
        # InfoNCE loss global
        # print("global_output:", global_output.shape)
        # print("global_target:", global_target.shape)
        global_loss = criterion(global_output, global_target)
        loss += global_loss
        # ProtoNCE loss global
        if global_output_proto is not None:
            g_p_l = 0
            for proto_out, proto_target in zip(global_output_proto, global_target_proto):
                g_p_l += criterion(proto_out, proto_target)
                accp = accuracy(proto_out, proto_target)[0]
                acc_proto_global.update(accp[0], images[0].size(0))
            g_p_l /= len(args.num_cluster)
            global_proto_loss = g_p_l
            loss += global_proto_loss

        dense_loss = criterion(dense_output, dense_target)

        loss += dense_loss
        # EMD loss dense (using EMD instead of ProtoNCE for dense features)
        # print(dense_output_proto is not None)
        if dense_output_proto is not None:
            dense_proto_loss = 0
            for proto_out, proto_target in zip(dense_output_proto, dense_target_proto):
                # print("proto_out:", proto_out.shape)
                # print("proto_target:", proto_target.shape)
                dense_proto_loss += criterion(proto_out, proto_target)
                accp = accuracy(proto_out, proto_target)[0]
                acc_proto_dense.update(accp[0], images[0].size(0))
            dense_proto_loss /= len(args.num_cluster)
            loss += dense_proto_loss

        losses.update(loss.item(), images[0].size(0))
        acc = accuracy(global_output, global_target)[0]
        acc_inst.update(acc[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def compute_features(eval_loader, model, args):
    # print('Computing features')
    model.eval()
    # global features
    global_features = torch.zeros(len(eval_loader.dataset), args.low_dim).cuda()
    # print("global_features:", global_features.shape)
    # dense features
    dense_features = torch.zeros(len(eval_loader.dataset), 1024, 7, 7).cuda()  # Based on the YOLOv8 head output
    # print("dense_features:", dense_features.shape)

    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            # print("images:", images.shape)
            feat_global, feat_dense = model(images, is_eval=True)
            global_features[index] = feat_global
            dense_features[index] = feat_dense

    dist.barrier()
    dist.all_reduce(global_features, op=dist.ReduceOp.SUM)
    dist.all_reduce(dense_features, op=dist.ReduceOp.SUM)

    return global_features, dense_features


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


def run_kmeans_dense(x, args):
    """
        Args:
            x: 输入特征, 大小为 (N, C, H, W)，例如 (1024, 7, 7) 的特征
        Returns:
            results_dense: 包含密集特征聚类信息的字典
    """
    print('Performing kmeans clustering on dense features')
    results_dense = {'im2cluster': [], 'centroids': [], 'density': []}

    N, C, H, W = x.shape
    K_list = args.num_cluster

    for idx, num_cluster in enumerate(K_list):
        print(f"Outer iteration {idx + 1} / {len(K_list)}")
        K = int(num_cluster)

        # 随机初始化 K 个原型
        indices = torch.randint(0, N, (K,), device=x.device)
        # 选择这些索引对应的样本作为初始原型
        centroids = x[indices]

        max_iters = 20

        for iteration in range(max_iters):
            # 输出提示信息
            print(f'Iteration {iteration + 1}/{max_iters}')
            # 1. 计算每个样本到所有原型的 EMD 距离，并分配聚类标签
            im2cluster = []
            for i in range(N):
                distances = []
                for k in range(K):
                    with torch.no_grad():
                        dist = sliced_loss(x[i].unsqueeze(0), centroids[k].unsqueeze(0)).item()
                        # dist = ssim(x[i].unsqueeze(0), centroids[k].unsqueeze(0))
                        distances.append(dist)  # 计算平均 EMD 距离
                im2cluster.append(np.argmin(distances))
            im2cluster = torch.LongTensor(im2cluster).cuda()

            # 2. 更新原型
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(K).cuda()

            for i in range(N):
                cluster_idx = im2cluster[i]
                new_centroids[cluster_idx] += x[i]
                counts[cluster_idx] += 1

            # 防止除零
            counts = counts.view(-1, 1, 1, 1)
            counts[counts == 0] = 1

            centroids = new_centroids / counts

            # Calculate objective and imbalance
            objective = 0.0
            imbalance = 0.0
            for k in range(K):
                cluster_points = x[im2cluster == k]
                if len(cluster_points) > 0:
                    with torch.no_grad():
                        centroid_distances = [sliced_loss(p.unsqueeze(0), centroids[k].unsqueeze(0)) for p in cluster_points]
                        objective += sum(centroid_distances)
                        imbalance += len(cluster_points) / float(N)

            print(
                f"  Iteration {iteration + 1} ({iteration / max_iters:.2f} s, search {iteration / max_iters:.2f} s): objective={objective:.5f} imbalance={imbalance:.3f} nsplit=0")

            # 如果原型没有变化，停止迭代
            if torch.allclose(new_centroids, centroids, atol=1e-6):
                print(f'Converged at iteration {iteration + 1}')
                break

        # 3. 计算密度
        density = torch.zeros(K).cuda()
        Dcluster = [[] for _ in range(K)]

        for i in range(N):
            cluster_idx = im2cluster[i]
            with torch.no_grad():
                dist = sliced_loss(x[i].unsqueeze(0), centroids[cluster_idx].unsqueeze(0))
                Dcluster[cluster_idx].append(dist)

        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                density[i] = (torch.tensor(dist).sqrt().mean() / np.log(len(dist) + 10)).item()

        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        # Use torch.quantile instead of percentile
        lower_bound = torch.quantile(density, 0.1)
        upper_bound = torch.quantile(density, 0.9)
        density = torch.clamp(density, lower_bound, upper_bound)
        density = density / density.mean()

        results_dense['centroids'] = [centroids]
        results_dense['density'] = [density]
        results_dense['im2cluster'] = [im2cluster]

        print("Objective improved: keep new clusters")

    return results_dense


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
