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

    print('===> Parameter Number:', utils.get_parameter_number(model_without_ddp))

    cp_path = f'./checkpoints/{args.dataset}/{args.file_name}'

    train_data = get_loader(args, 'train')

    if args.resume or args.test_only or args.pre_trained:
        model_path = args.load_name if os.path.exists(args.load_name) else f"{cp_path}/{args.load_name}"
        try:
            checkpoint = torch.load(model_path, map_location='cuda:{}'.format(args.local_rank))
            model_without_ddp.load_state_dict(checkpoint['model'])
            if args.resume:
                args.start_epoch = checkpoint['epoch'] + 1
                lr_scheduler.step(args.start_epoch)
                # optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            print('===> File {} not exists'.format(model_path))
        else:
            print('===> File {} loaded'.format(model_path))
        evaluate(model, criterion, args.test_name, device=device, val_data=get_loader(args, args.test_name), args=args)



if __name__ == '__main__':

    main()










