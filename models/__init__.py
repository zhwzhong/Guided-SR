# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   __init__.py.py
@Time    :   2023/2/1 19:39
@Desc    :
"""
from importlib import import_module

def get_model(args):
    module = import_module('models.' + args.model_name.lower())
    return module.make_model(args)