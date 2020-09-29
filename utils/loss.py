import torch
from torch import nn
class L2_loss_func(nn.Module):
    def __init__(self):
        super(L2_loss_func, self).__init__()

    def forward(self, x, target):
        N = x.shape[0]
        a = (target[:, 0, :, :] == 1).unsqueeze(dim=1)
        b = (target[:, 0, :, :] == 0).unsqueeze(dim=1)
        loss_pt = (x[:, :, :, :] - target[:, :, :, :]) ** 2 * a * 0.75
        loss_nopt = (x[:, 0, :, :].view(N,1,14,14) - target[:, 0, :, :].view(N,1,14,14)) ** 2 * b * 0.25
        loss = loss_pt.sum(dtype=float) + loss_nopt.sum(dtype=float)
        return loss