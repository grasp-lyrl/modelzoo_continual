#!/usr/bin/env python3
"""
Helper function to initialize a neural network
"""
from typing import Any

from net.wideresnet import WideResNetMultiTask
from net.smallconv import SmallConv


def fetch_net(args: Any,
              num_tasks: int,
              num_cls: int,
              dropout: float = 0.3):
    """
    Create a nearal network to train
    """
    if "mnist" in args.dataset:
        inp_chan = 1
        pool = 2
        l_size = 80
    elif args.dataset == "mini_imagenet":
        inp_chan = 3
        pool = 3
        l_size = 320
    elif "cifar" in args.dataset:
        inp_chan = 3
        pool = 2
        l_size = 320
    else:
        raise NotImplementedError

    if args.model == "wrn16_4":
        net = WideResNetMultiTask(depth=16, num_task=num_tasks,
                                  num_cls=num_cls, widen_factor=4,
                                  drop_rate=dropout, inp_channels=inp_chan)
    elif args.model == "conv":
        net = SmallConv(num_task=num_tasks, num_cls=num_cls,
                        channels=inp_chan, avg_pool=pool,
                        lin_size=l_size)
    else:
        raise ValueError("Invalid network")

    if args.gpu:
        net.cuda()
    return net
