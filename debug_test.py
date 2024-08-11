import torch

if __name__ == "__main__":
    l_dense_pos = [1., 2., 3, 4]
    # l_dense_pos = torch.stack(l_dense_pos, dim=0)  # (N)
    l_dense_pos = torch.tensor(l_dense_pos)  # (N)
    print(l_dense_pos.shape)
    print(l_dense_pos)