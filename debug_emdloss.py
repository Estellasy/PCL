import torch
import torch.nn.functional as F

class EMDLoss(torch.nn.Module):
    def __init__(self, lambda_reg=1.0, num_iterations=10):
        super(EMDLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.num_iterations = num_iterations

    def forward(self, X, Y):
        # Ensure X and Y have the same shape
        assert X.shape == Y.shape
        B, C, H, W = X.shape
        HW = H * W

        # Flatten feature maps
        X_flat = X.view(B, C, HW)  # Shape: (B, C, HW)
        Y_flat = Y.view(B, C, HW)  # Shape: (B, C, HW)

        # Compute cost matrix
        X_norm = F.normalize(X_flat, p=2, dim=1)  # Shape: (B, HW, C)
        Y_norm = F.normalize(Y_flat, p=2, dim=1)  # Shape: (B, HW, C)
        M = 1 - torch.matmul(X_norm.transpose(1, 2), Y_norm)  # Shape: (B, HW, HW)

        # Initialize marginal weights
        r = torch.ones(B, HW, device=X.device) / HW
        c = torch.ones(B, HW, device=X.device) / HW

        # Sinkhorn-Knopp algorithm
        for _ in range(self.num_iterations):
            u = 1.0 / (torch.matmul(M, c.unsqueeze(2)).squeeze(2) + 1e-8)  # Shape: (B, HW)
            v = 1.0 / (torch.matmul(M.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)  # Shape: (B, HW)
            u = r / (torch.matmul(M, v.unsqueeze(2)).squeeze(2) + 1e-8)  # Shape: (B, HW)
            v = c / (torch.matmul(M.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)  # Shape: (B, HW)

        # Optimal transport plan
        P = torch.exp(-self.lambda_reg * M)  # Shape: (B, HW, HW)
        P = torch.matmul(torch.diag_embed(u), torch.matmul(P, torch.diag_embed(v)))  # Shape: (B, HW, HW)

        # Compute similarity score
        similarity = torch.sum(P * (1 - M), dim=[1, 2])

        # Compute EMD loss
        L_EMD = 2 - 2 * similarity

        return M, similarity, L_EMD.mean()

# Example usage
if __name__ == "__main__":
    batch_size = 1
    channels = 1024
    height = width = 7

    # Random feature maps
    X = torch.randn(batch_size, channels, height, width)
    Y = torch.randn(batch_size, channels, height, width)

    # EMD Loss
    emd_loss_fn = EMDLoss(lambda_reg=1.0, num_iterations=100)
    M, similarity, loss = emd_loss_fn(X, Y)
    print(M)
    print("simlarity:", similarity)
    print("EMD Loss:", loss.item())
    
    M, similarity, loss = emd_loss_fn(X, X)
    print(M)
    print("simlarity:", similarity)
    print("EMD Loss:", loss.item())
    
    similarity, loss = emd_loss_fn(Y, Y)
    print("simlarity:", similarity)
    print("EMD Loss:", loss.item())
