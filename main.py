# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   main.py
@Time    :   2022/3/1 20:06
@Desc    :
"""

import os
import json
import loss
import utils
import torch
from options import args
from data import get_loader
from models import get_model
from scheduler import create_scheduler
from trainer import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
from utils import make_optimizer, set_checkpoint_dir, master_only

def main():
    model = get_model(args)
    set_checkpoint_dir(args)
    device = torch.device(args.device)

    writer = SummaryWriter('./logs/{}/{}'.format(args.dataset, args.file_name))
    model.to(device)
    criterion = loss.Loss(args)
    optimizer = make_optimizer(args, model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    else:
        model = torch.nn.parallel.DataParallel(model, device_ids=list(range(args.num_gpus)))
    model_without_ddp = model.module

        evaluate(model, criterion, args.test_name, device=device, val_data=get_loader(args, args.test_name), args=args)



if __name__ == '__main__':

    main()










