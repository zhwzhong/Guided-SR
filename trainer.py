# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   trainer.py
@Time    :   2022/3/1 20:06
@Desc    :
"""
import os
import cv2
import numpy as np

import utils
import torch
import tqdm

def train_one_epoch(model, criterion, train_data, optimizer, device, epoch, args):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for samples in metric_logger.log_every(train_data, args.print_freq, header):
        samples = utils.to_device(samples, device)
        out = model(utils.mix_up(samples, args.mix_alpha) if args.mix_up else samples)
        loss = criterion(out['img_out'], samples['img_gt'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, test_name, val_data, device, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    sv_path = './track1/evaluation1/x4/' if args.scale == 4 else './track1/evaluation2/x2/'
    if args.dataset == 'NIR':
        sv_path = './track2/evaluation/x8/'

    if args.save_result and args.local_rank == 0:
        utils.create_dir(sv_path)

    nb = len(val_data)
    val_data = enumerate(val_data)
    if args.local_rank in [-1, 0]:
        val_data = tqdm.tqdm(val_data, total=nb)  # 只在主进程打印进度条

    for _, samples in val_data:
        samples = utils.to_device(samples, device)
        out = utils.ensemble(samples, model, args.ensemble_mode, args.dataset) if args.self_ensemble else model(samples)
        torch.cuda.synchronize()
        loss = criterion(out['img_out'], samples['img_gt'])
        metric_logger.update(loss=loss.item() * 1000)

        if args.save_result:
            for index in range(samples['img_gt'].size(0)):
                save_name = os.path.join(sv_path, samples['img_name'][0])
                img = utils.tensor2uint(out['img_out'][index: index + 1], data_range=args.data_range)
                if args.dataset == 'NIR':
                    cv2.imwrite(save_name, img)
                    np.save(save_name.replace('bmp', 'npy'), out['img_out'][index: index + 1].detach().cpu().numpy())
                else:
                    cv2.imwrite(save_name.replace('npy', 'jpg'), img, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
                print('Image Saved to {}'.format(save_name))
        metrics = utils.calc_metrics(out['img_out'], samples['img_gt'], args)
        # print(metrics, samples['img_gt'].size(0), os.path.join(sv_path, samples['img_name'][0]))
        for metric, value in metrics.items():
            metric_logger.meters[metric].update(value.item(), n=samples['img_gt'].size(0))

    metric_logger.synchronize_between_processes()
    torch.cuda.empty_cache()
    metric_out = {'{}_'.format(test_name) + k: round(meter.global_avg, 3) for k, meter in metric_logger.meters.items()}
    print(metric_out)
    return metric_out