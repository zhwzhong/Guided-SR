# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   set_config.py
@Time    :   2023/2/3 20:08
@Desc    :
"""


def set_config(args):

    if args.sched == 'multistep':
        args.decay_epochs = [int(x) for x in args.decay_epochs.split('_')]
    else:
        args.decay_epochs = int(args.decay_epochs)

    if args.file_name == '':
        args.file_name = args.model_name
    args.file_name += f'_S_{args.scale}_Loss_{args.loss}_LR_{args.lr}_Bs_{args.batch_size}_Ps_{args.patch_size}_Sed_{args.seed}'

    if args.rgb_norm:
        args.file_name += '_Norm'
    if args.mix_up:
        args.file_name += f'_MX_{args.mix_alpha}'

    if args.pre_trained:
        args.file_name += '_PT'

    if args.extra_data:
        args.file_name += '_ED'

    if args.no_res:
        args.file_name += '_NR'

    if args.with_noisy:
        args.file_name += '_Noisy'

    if args.random_down:
        args.file_name += '_RD'

    if args.mat_resize:
        args.file_name += '_MR'

    if args.local_rank == 0:
        print(args)