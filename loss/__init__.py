# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   __init__.py.py
@Time    :   2022/3/1 09:55
@Desc    :
"""
import torch
from torch import nn
from .BALoss import BALoss
from .CharbonnierLoss import CharbonnierLoss
from .image_gradient_loss import GradientSensitiveLoss


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_func = nn.MSELoss(reduction='mean')
            elif loss_type == 'L1':
                loss_func = nn.L1Loss(reduction='mean')
            elif loss_type == 'Huber':
                loss_func = nn.HuberLoss(reduction='mean')
            elif loss_type == 'CharbonnierLoss':
                loss_func = L1_Charbonnier_loss()
            elif loss_type == 'SmoothL1':
                loss_func = nn.SmoothL1Loss(reduction='mean')
            else:
                raise NotImplementedError
            self.loss.append({'type': loss_type, 'weight': float(weight), 'function': loss_func})

        for l in self.loss:
            if args.local_rank == 0:
                print('Loss Function: {:.3f} * {}'.format(l['weight'], l['type']))
            self.loss_module.append(l['function'])

        device = torch.device(args.device)
        self.loss_module.to(device)

    def forward(self, out, gt, mask=None):
        losses = []

        for i, l in enumerate(self.loss):
            if mask is None:
                loss = l['function'](out, gt)
            else:
                loss = l['function'](out[mask == 1.], gt[mask == 1.])
            effective_loss = l['weight'] * loss
            losses.append(effective_loss)
        return sum(losses)

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
