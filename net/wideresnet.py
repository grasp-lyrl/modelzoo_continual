#!/usr/bin/env python3
"""
Implementation of Wide-Resnets for Multi-task learning.
Adapted from an open-source implementation.
https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch import Tensor, Type


class BasicBlock(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 stride: int,
                 drop_rate: float = 0.0) -> None:
        nn.Module.__init__(self)
        # super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = nn.Conv2d(in_planes, out_planes,
                                      kernel_size=1, stride=stride,
                                      padding=0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self,
                 nb_layers: int,
                 in_planes: int,
                 out_planes: int,
                 block: BasicBlock,
                 stride: int,
                 drop_rate: float = 0.0) -> None:
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes,
                                      nb_layers, stride, drop_rate)

    def _make_layer(self,
                    block: BasicBlock,
                    in_planes: int,
                    out_planes: int,
                    nb_layers: int,
                    stride: int,
                    drop_rate: float) -> nn.Sequential:
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class WideResNetMultiTask(nn.Module):
    """
    Wide-Resnet (https://arxiv.org/abs/1605.07146) for multiple tasks.
    This implementation assumes all tasks have the same number of classes. 
    See WideResNetMultiTask_v2 if you want to have multiple classes
    """
    def __init__(self,
                 depth: int,
                 num_task: int,
                 num_cls: int,
                 widen_factor: int = 1,
                 drop_rate:float = 0.0,
                 inp_channels: int = 3) -> None:
        """
        Args:
            - depth: Depth of WRN
            - num_task: Number of tasks (number of classification layers)
            - num_cls: Number of classes for each task (same for all tasks)
            - widen_factor: The scaling factor for the number of channels
            - drop_rate: Dropout prob. that element is zeroed out
            - inp_channels: Number of channels in the input
        """
        super(WideResNetMultiTask, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        self.conv1 = nn.Conv2d(inp_channels, nChannels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.block1 = NetworkBlock(n, nChannels[0],
                                   nChannels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1],
                                   nChannels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2],
                                   nChannels[3], block, 2, drop_rate)

        # global average pooling to accomodate flexible input image sizes
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Multiple linear layers, one for each task
        lin_layers = []
        for task in range(num_task):
            lin_layers.append(nn.Linear(nChannels[3], num_cls))

        self.fc = nn.ModuleList(lin_layers)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self,
                x: Tensor,
                tasks: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.pool(out)
        out = out.view(-1, self.nChannels)

        # Fill logits with zeros
        logits = self.fc[0](out) * 0

        # Get the output of the linear layer corresponding to the task
        for idx, lin in enumerate(self.fc):
            task_ind = idx == tasks
            task_idx = torch.nonzero(task_ind, as_tuple=False).view(-1)
            if len(task_idx) == 0:
                continue

            task_out = torch.index_select(out, dim=0, index=task_idx)
            task_logit = lin(task_out)
            logits.index_add_(0, task_idx, task_logit)

        return logits


class WideResNetMultiTask_v2(nn.Module):
    """
    This function is not used in the code but included for completeness.
    Multi-task implementation of WRN where all tasks don't have the same number
    of classes
    """
    def __init__(self,
                 depth: int,
                 task_outputs: List[int],
                 widen_factor: int = 1,
                 drop_rate: float = 0.0,
                 inp_channels: int = 3,
                 use_gpu: bool = True) -> None:
        """
        Args: 
            - depth: Depth of WRN
            - task_outputs: A list containing number of outputs for each tasks
            - widen_factor: The scaling factor for the number of channels
            - drop_rate: Dropout prob. that element is zeroed out
            - inp_channels: Number of channels in the input
        """

        super(WideResNetMultiTask_v2, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        self.gpu = use_gpu

        # Get task_id with most number of classes
        self.max_id = -1
        maxval = -1
        for t_id, out_size in enumerate(task_outputs):
            maxval = max(maxval, out_size)
            if maxval == out_size:
                self.max_id = t_id

        # Get padding required for each task
        self.pad_map = {}
        for t_id, out_size in enumerate(task_outputs):
            self.pad_map[t_id] = maxval - out_size

        self.conv1 = nn.Conv2d(inp_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0],
                                   nChannels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1],
                                   nChannels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2],
                                   nChannels[3], block, 2, drop_rate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        lin_layers = []

        # Classification layers with outputs of different sizes
        for out_size in task_outputs:
            numlabs = out_size
            lin_layers.append(nn.Linear(nChannels[3], numlabs))

        self.fc = nn.ModuleList(lin_layers)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self,
                x: Tensor,
                tasks: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.pool(out)
        out = out.view(-1, self.nChannels)

        # Fill logits with zeros
        logits = self.fc[self.max_id](out) * 0
        max_shape = logits.size()

        # Fill out the logits for the entire batch
        for idx, lin in enumerate(self.fc):
            task_ind = idx == tasks
            task_idx = torch.nonzero(task_ind, as_tuple=False).view(-1)
            if len(task_idx) == 0:
                continue

            task_out = torch.index_select(out, dim=0, index=task_idx)
            task_logit = lin(task_out)
            shape = task_logit.size()
            if self.pad_map[idx] > 0:
                pad = torch.zeros((len(task_logit), self.pad_map[idx]))
                if self.gpu:
                    pad = pad.cuda()
                task_logit = torch.cat([task_logit, pad], dim=1)

            logits.index_add_(0, task_idx, task_logit)

        return logits
