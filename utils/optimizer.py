# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   optimizer.py
@Time    :   2022/8/13 16:45
@Desc    :
"""
import torch.optim as optim

def make_optimizer(args, targets):

    if args.opt == 'AMSGrad':
        optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    elif args.opt == 'AdamW':
        optimizer = optim.AdamW(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optimizer