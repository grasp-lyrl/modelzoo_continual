#!/usr/bin/env python3
"""
Helper function to initialize a neural network
"""
import torch
from net.wideresnet import WideResNetMultiTask
from net.smallconv import SmallConv
from typing import Any


def fetch_net(args: Any,
              num_tasks: int,
              num_cls: int,
              dropout: float = 0.3):
    """
    Create a nearal network to train
    """
    if "mnist" in args.dataset:
        inp_chan = 1
    else:
        inp_chan = 3

    if args.model == "wrn16_4":
        net = WideResNetMultiTask(depth=16, num_task=num_tasks,
                                  num_cls=num_cls, widen_factor=4,
                                  drop_rate=dropout, inp_channels=inp_chan)
    elif args.model == "conv":
        if args.dataset == "mini_imagenet":
            pool = 3
        else:
            pool = 2
        net = SmallConv(num_task=num_tasks, num_cls=num_cls,
                        channels=inp_chan, avg_pool=pool)
    else:
        raise ValueError("Invalid network")

    if args.gpu:
        net.cuda()
    return net
