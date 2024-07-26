"""
@Description :
@Author      : siyiren1@foxmail.com
@Time        : 2024/07/25 22:06:14
"""
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

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
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
        args.low_dim, args.pcl_r,  args.moco_m, args.temperature, args.mlp
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
    emd_loss_fn = EMDLoss(eps=0.1, max_iter=100)

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
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size*5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)
    
    # 修改训练epoch逻辑
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
            for num_cluster in args.num_cluster:
                cluster_result_dense['im2cluster'].append(torch.zeros(len(eval_dataset), dtype=torch.long).cuda())
                cluster_result_dense['centroids'].append(torch.zeros(int(num_cluster), 1024 * 20 * 20).cuda())  # Flattened size
                cluster_result_dense['density'].append(torch.zeros(int(num_cluster)).cuda())

            if args.gpu == 0:
                global_features[torch.norm(global_features,dim=1)>1.5] /= 2 # account for the few samples that are computed twice  
                global_features = global_features.numpy()
                cluster_result_global = run_kmeans(global_features, args)
                print("global")
                # dense
                dense_features = dense_features.view(dense_features.size(0), -1)  # Flatten dense features
                dense_features[dense_features.norm(dim=1) > 1.5] /= 2
                dense_features = dense_features.numpy()
                cluster_result_dense = run_kmeans(dense_features, args)
            
                print("dense")
            dist.barrier()  
            print("dist bug")
            # broadcast clustering result
            for k, data_list in cluster_result_global.items():
                for data_tensor in data_list:                
                    dist.broadcast(data_tensor, 0, async_op=False)     

                print("broadcast global")

            for k, data_list in cluster_result_dense.items():
                for data_tensor in data_list:                
                    dist.broadcast(data_tensor, 0, async_op=False)  

                print("broadcast dense")   

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, emd_loss_fn, optimizer, epoch, args, cluster_result_global, cluster_result_dense)

        if (epoch+1)%5==0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir,epoch))

def train(train_loader, model, criterion, emd_loss_fn, optimizer, epoch, args, cluster_result_global=None, cluster_result_dense=None):
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
        [batch_time, data_time, losses, global_loss, dense_loss, global_proto_loss, dense_proto_loss, acc_inst, acc_proto_global, acc_proto_dense],
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
        print("global_output:", global_output.shape)
        print("global_target:", global_target.shape)
        global_loss = criterion(global_output, global_target)
        loss += global_loss
        # ProtoNCE loss global
        if global_output_proto is not None:
            global_proto_loss = 0
            for proto_out, proto_target in zip(global_output_proto, global_target_proto):
                global_proto_loss += criterion(proto_out, proto_target)
                accp = accuracy(proto_out, proto_target)[0]
                acc_proto_global.update(accp[0], images[0].size(0))
            global_proto_loss /= len(args.num_cluster)
            loss += global_proto_loss

        print("dense_output:", dense_output.shape)
        print("dense_target:", dense_target.shape)
        # 将 dense_output 展平成 (N, C) 形状
        N, D, C = dense_output.size()
        dense_output_flattened = dense_output.view(N * D, C)

        # 扩展 dense_target 以匹配 dense_output_flattened 的第一个维度
        dense_target_expanded = dense_target.unsqueeze(1).expand(-1, D).flatten()
        dense_loss = criterion(dense_output_flattened, dense_target_expanded)
        loss += dense_loss
        # EMD loss dense (using EMD instead of ProtoNCE for dense features)
        print(dense_output_proto is not None)
        if dense_output_proto is not None:
            dense_proto_loss = 0
            for proto_out, proto_target in zip(dense_output_proto, dense_target_proto):
                print("proto_out:", proto_out.shape)
                print("proto_target:", proto_target.shape)
                dense_proto_loss += emd_loss_fn(proto_out, proto_target)
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
    print('Computing features')
    model.eval()
    # global features 
    global_features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda()
    print("global_features:", global_features.shape)
    # dense features
    dense_features = torch.zeros(len(eval_loader.dataset), 1024, 7, 7).cuda()  # Based on the YOLOv8 head output
    print("dense_features:", dense_features.shape)

    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            print("images:", images.shape)
            feat_global, feat_dense = model(images, is_eval=True)
            global_features[index] = feat_global
            dense_features[index] = feat_dense

    dist.barrier()
    dist.all_reduce(global_features, op=dist.ReduceOp.SUM)   
    dist.all_reduce(dense_features, op=dist.ReduceOp.SUM)

    return global_features.cpu(), dense_features.cpu()


def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
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
        
        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results 


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
