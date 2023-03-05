# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   image_gradient_loss.py
@Time    :   2019/11/4 11:04
@Desc    :
"""
import torch
import torch.nn as nn
from loss.gaussian import GaussianSmoothing

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.gradient_loss_func = 'L1'
        if self.gradient_loss_func == 'L1':
            self.criterion = nn.L1Loss(reduction='mean')
        else:
            self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        output_gradient_x, output_gradient_y = get_gradient(output)
        target_gradient_x, target_gradient_y = get_gradient(target)
        loss = self.criterion(output_gradient_x, target_gradient_x) + self.criterion(output_gradient_y, target_gradient_y)
        return loss / 2.

def get_gradient(y):
    right_y = y[:, :, 1:, 0: -1]
    down_y = y[:, :, 0: -1, 1: ]
    clip_y = y[:, :, 0: -1, 0: -1]
    gradient_h = torch.abs(right_y - clip_y)
    gradient_y = torch.abs(down_y - clip_y)
    return gradient_h, gradient_y


class GradientSensitiveLoss(nn.Module):
    def __init__(self, channels, trade_off=0.5, gradient_loss_func='L1', dim=2):
        super(GradientSensitiveLoss, self).__init__()
        self.trade_off = trade_off
        if gradient_loss_func == 'L1':
            self.criterion = nn.L1Loss(reduction='mean')
        else:
            self.criterion = nn.MSELoss(reduction='mean')

        self.gaussian = GaussianSmoothing(channels=channels, kernel_size=15, sigma=5, dim=dim)

    def forward(self, output, target):
        out_low, gt_low = self.gaussian(output), self.gaussian(target)
        out_high, gt_high = output - out_low, target - gt_low
        return self.criterion(out_low, gt_low) + self.trade_off * self.criterion(out_high, gt_high)