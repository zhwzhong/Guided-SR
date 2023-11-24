# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   nir.py
@Time    :   2023/2/1 20:01
@Desc    :
"""

import os
import cv2
import h5py
import numpy as np
from data import augment
from torch.utils.data import Dataset
from utils.image_resize import imresize


root_path = None
for path in ['/data_c/zhwzhong', '/home/temp', '/home/zhwzhong']:
    tmp_path = '{}/Data'.format(path)
    if os.path.exists(tmp_path):
        root_path = tmp_path


class NIR(Dataset):
    def __init__(self, args, attr):
        self.args = args
        self.attr = attr
        self.file = h5py.File('{}/PBVS/track2/guided_nir.h5'.format(root_path))[attr]

        func = lambda x: np.array(x) if self.args.cached and attr == 'train' else x

    def __len__(self):
        return int(self.args.show_every * len(self.gt_imgs)) if self.attr == 'train' else len(self.gt_imgs)

    def __getitem__(self, item):
        item = item % len(self.gt_imgs)

        gt_img, rgb_img = np.array(self.gt_imgs[item]), np.array(self.rgb_imgs[item])
        gt_img, rgb_img = np.expand_dims(gt_img, 0), np.transpose(rgb_img, (2, 0, 1))

        if self.attr == 'train':
            gt_img, rgb_img = augment.get_patch(gt_img, rgb_img, patch_size=self.args.patch_size)
            gt_img, rgb_img = augment.random_rot(gt_img, rgb_img, hflip=True, rot=True)

        if self.attr == 'test':
            lr_img = gt_img / 255
        else:
            if self.args.mat_resize:
                lr_img = imresize(gt_img.squeeze(), scalar_scale=1 / self.args.scale)
            else:
                lr_img = cv2.resize(gt_img.squeeze(), fx=1 / self.args.scale, fy=1 / self.args.scale, dsize=None)
            lr_img = imresize(lr_img.astype(float), scalar_scale=self.args.scale) / 255
            lr_img = np.expand_dims(lr_img, 0)

        gt_img, rgb_img = gt_img / 255, rgb_img / 255

        if self.args.rgb_norm:
            i_std = np.array([0.22773795, 0.22367531, 0.26343636]).reshape(3, 1, 1)
            i_mean = np.array([0.45617331, 0.49676442, 0.44097971]).reshape(3, 1, 1)
            rgb_img = (rgb_img - i_mean) / i_std

        lr_img, gt_img, rgb_img = augment.np_to_tensor(lr_img, gt_img, rgb_img, input_data_range=1)

        return {'img_gt': gt_img, 'img_rgb': rgb_img, 'lr_up': lr_img, 'img_name': self.img_names[item]}

# from options import args
# from utils import metrics
# from torch.utils.data import DataLoader
# from torch.nn.functional import interpolate
# args.scale = 4
# args.extra_data = True
# batch_size = 1
# nyu_data = DataLoader(NIR(args, 'train'), batch_size=batch_size)
# print(len(nyu_data))
# sum_psnr = 0
# for _, sample in enumerate(nyu_data):
#     print(sample['img_gt'].shape, sample['img_name'], sample['lr_up'].shape)
#     sum_psnr += metrics(sample['img_gt'], sample['lr_up'], 0, 1)['PSNR']
#
# print(sum_psnr / 9)
