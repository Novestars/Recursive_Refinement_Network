#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/9/28 下午12:13
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : init_func.py.py
import torch
import torch.nn as nn


def __init_weight(feature, conv_init, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
            conv_init(m.weight, **kwargs)

def __init_weight_batch(feature, conv_init, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.GroupNorm)):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)


def init_weight_batch(module_list, conv_init, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight_batch(feature[0], conv_init, **kwargs)
    else:
        __init_weight_batch(module_list, conv_init, **kwargs)
def init_weight(module_list, conv_init, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature[0], conv_init, **kwargs)
    else:
        __init_weight(module_list, conv_init, **kwargs)



def group_weight(weight_group, module, lr, no_decay_lr=None):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)) or isinstance(m, (
        nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    lr = lr if no_decay_lr is None else no_decay_lr
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group
