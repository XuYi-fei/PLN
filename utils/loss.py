import torch
from torch import nn
class L2_loss_func(nn.Module):
    def __init__(self):
        super(L2_loss_func, self).__init__()

    def forward(self, x, target):
        print(target[0,21:23,:,:])
        N = x.shape[0]
        a = (target[:, 0, :, :] == 1).unsqueeze(dim=1)
        b = (target[:, 0, :, :] == 0).unsqueeze(dim=1)
        w_classes = 0.2
        w_L = 0.1
        w_xy = 0.1
        loss_p =(x[:, 0, :, :].unsqueeze(dim=1) - target[:, 0, :, :].unsqueeze(dim=1)) ** 2 * a
        loss_classes = (x[:, 1:21, :, :] - target[:, 1:21, :, :]) ** 2 * w_classes * a
        loss_xy = (x[:, 21:23, :, :] - target[:, 21:23, :, :]) ** 2 * w_xy * a
        loss_lin = (x[:, 23:, :, :] - target[:, 23:, :, :]) ** 2 * a * w_L
        loss_pt = (loss_p.sum(dtype=float) + loss_xy.sum(dtype=float) + loss_lin.sum(dtype=float) + loss_classes.sum(dtype=float))
        loss_nopt = (x[:, 0, :, :].view(N,1,14,14) - target[:, 0, :, :].view(N,1,14,14)) ** 2 * b

        loss = loss_nopt.sum(dtype=float) + loss_pt



        return loss