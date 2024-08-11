import torch, sys
"""
Implementation of the Sinkhorn-Knopp algorithm described in
[1] Sinkhorn Distances: Lightspeed Computation of Optimal Transport
"""


def sinkhorn(r, c, M, reg=1e-3, error_thres=1e-5, num_iters=200):
    """Batch sinkhorn iteration. See a blog post <https://kaizhao.net/blog/optimal-transport> (in Chinese) for explainations.

    Args:
        r (torch.tensor): tensor with shape (n, d1), the first distribution .
        c (torch.tensor): tensor with shape (n, d2), the second distribution.
        M (torch.tensor): tensor with shape (n, d1, d2) the ground metric.
        reg (float): factor for entropy regularization (corresponds to \frac{1}{\lambda} in [1]).
        error_thres (float): the error threshold to stop the iteration. 
        num_iters (int): number of total iterations.

    Returns:
        torch.tensor: the optimal transportation matrix (n, d1, d2).
    """
    n, d1, d2 = M.shape
    assert r.shape[0] == c.shape[0] == n and \
           c.shape[1] == d1 and c.shape[1] == d2, \
           'r.shape=%s, v=shape=%s, M.shape=%s' % (r.shape, c.shape, M.shape)

    K = (-M / reg).exp()        # (n, d1, d2)
    u = torch.ones_like(r) / d1 # (n, d1)
    v = torch.ones_like(c) / d2 # (n, d2)

    for _ in range(num_iters):
        r0 = u
        # u = r / K \cdot v
        u = r / torch.einsum('ijk,ik->ij', [K, v])
        # v = c / K^T \cdot u
        v = c / torch.einsum('ikj,ik->ij', [K, u])

        err = (u - r0).abs().mean()
        if err.item() < error_thres:
            break
    T = torch.einsum('ij,ik->ijk', [u, v]) * K
    return T, u, v


def log_sinkhorn(u, v, M, reg=1e-5, error_thres=1e-5, num_iters=100):
    raise NotImplemented


class SinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, M, reg, error_thres=1e-5, num_iters=100, grad_shift=True):
        T, u, v = sinkhorn(u, v, M, reg, error_thres, num_iters)
        ctx.u = u
        ctx.v = v
        ctx.reg = reg
        ctx.K = (-M / reg).exp()
        ctx.grad_shift = grad_shift
        return (T * M).sum(dim=1).mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        logu = ctx.u.log()
        logv = ctx.v.log()

        gradu = ctx.reg * logu
        gradv = ctx.reg * logv

        if ctx.grad_shift:
            gradu -= (ctx.reg * logu.sum(dim=1).view(-1, 1, 1) / ctx.K).sum(dim=2)
            gradv -= (ctx.reg * logv.sum(dim=1).view(-1, 1, 1) / ctx.K).sum(dim=1)
        return gradu, gradv, None, None, None, None