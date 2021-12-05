#!/usr/bin/env python3
"""
Implementation of the Multihead learner, which trains a
single model with a shared network backbone, and
task-specific linear layers.
"""
import numpy as np
import torch
import torch.optim as optim
import torch.cuda.amp as amp

from net.build_net import fetch_net
from datasets.build_dataset import fetch_dataclass
from utils.logger import Logger
from utils.run_net import evaluate, run_epoch


class MultiHead():
    """
    Object for initializing and training a multihead learner
    """
    def __init__(self, args, hp, data_conf):
        """
        Initialize multihead learner

        Params:
          - args:      Arguments from arg parse
          - hp:        JSON config file for hyper-parameters
          - data_conf: JSON config of dataset
        """
        self.args = args
        self.hp = hp

        num_tasks = len(data_conf['tasks'])
        num_classes = len(data_conf['tasks'][0])

        # Random seed
        torch.manual_seed(abs(args.seed))
        np.random.seed(abs(args.seed))

        # Network
        # Currently code assumes all tasks have same number of classes
        self.net = fetch_net(args, num_tasks, num_classes, hp['dropout'])

        # Get dataset
        dataclass = fetch_dataclass(data_conf['dataset'])
        dataset = dataclass(args, data_conf['tasks'], limited_replay=True)
        self.train_loader = dataset.get_data_loader(hp["batch"], 4, train=True)

        test_loaders = []
        alltrain_loaders = []
        for t_id in range(num_tasks):
            alltrain_loaders.append(
                dataset.get_task_data_loader(t_id, hp['batch'],
                                             4, train=True))
            test_loaders.append(
                dataset.get_task_data_loader(t_id, hp['batch'],
                                             4, train=False))

        # Logger
        self.logger = Logger(test_loaders, alltrain_loaders,
                             num_tasks, num_classes, args)

        # Loss and Optimizer
        self.scaler = amp.GradScaler(enabled=args.fp16)
        self.optimizer = optim.SGD(self.net.parameters(), lr=hp['lr'],
                                   momentum=0.9, nesterov=True,
                                   weight_decay=hp['l2_reg'])
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, args.epochs * len(self.train_loader))

    def train(self, log_interval=5):
        """
        Train the multi-task learner

        Params:
          - log_interval: frequency with which test-set is evaluated
        """
        # Evaluate before start of training:
        train_met = evaluate(self.net, self.train_loader, self.args.gpu)
        self.logger.log_metrics(self.net, train_met, -1)

        # Train multi-head model
        for epoch in range(self.args.epochs):

            train_met = run_epoch(self.net, self.args, self.optimizer,
                                  self.train_loader, self.lr_scheduler,
                                  self.scaler)
            if epoch == self.args.epochs - 1:
                self.logger.log_metrics(self.net, train_met, epoch, True)
            elif epoch % log_interval == 0:
                self.logger.log_metrics(self.net, train_met, epoch)
            else:
                self.logger.log_train(self.net, train_met, epoch)

        return self.net, self.logger.train_accs
