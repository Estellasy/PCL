q_grid = q_grid.permute(0, 2, 1)
q_grid = q_grid.reshape(-1, q_grid.size(2))
l_neg_dense = torch.einsum('nc,ck->nk', [q_grid,
self.queue2.clone().detach()])