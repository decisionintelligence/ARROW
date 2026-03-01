import torch
from torch import nn
from torch.nn import functional as F

def sum_off_diagonal(matrix):
    # matrix: L x N x N
    return matrix.sum() - torch.diagonal(matrix, dim1=1, dim2=2).sum()

def pairwise_cross_entropy(P: torch.Tensor, Q: torch.Tensor = None, eps: float = 1e-9) -> torch.Tensor:
    if Q is None:
        Q = P

    P = P.unsqueeze(2)   # (L, N, 1, C)
    Q = Q.unsqueeze(1)   # (L, 1, N, C)

    log_Q = torch.log(Q + eps)  # (L, 1, N, C)
    cross_entropy = torch.sum(P * log_Q, dim=-1)  # (L, N, N)

    return cross_entropy

def uniform_cross_entropy(p):
    p = p / p.sum(dim=1, keepdim=True)
    D = p.size(1)
    return -torch.sum(torch.log(p + 1e-8), dim=1) / D


class SP_MoE_loss(nn.Module):
    def __init__(self, eps=1e-6, alpha=1e-2, beta=1, w_aux_1=True, w_aux_2=True):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.w_aux_1 = w_aux_1
        self.w_aux_2 = w_aux_2

    def forward(self, noises):
        # noises: layers x num_time_interval x E
        layers = noises.shape[0]

        noises = F.softmax(noises, dim=-1)
        aux_loss_1 = sum_off_diagonal(pairwise_cross_entropy(noises)) / (2 * layers)
        
        noises = noises.mean(dim=1)
        aux_loss_2 = uniform_cross_entropy(noises).mean(dim=0)

        if self.w_aux_1 and self.w_aux_2:
            return self.alpha*(self.beta * aux_loss_1 + aux_loss_2)
        elif self.w_aux_1 and not self.w_aux_2:
            return self.alpha * self.beta * aux_loss_1
        elif not self.w_aux_1 and self.w_aux_2:
            return self.alpha * aux_loss_2
        else:
            return 0.0
