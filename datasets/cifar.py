#!/usr/bin/env python3
"""
CIFAR10 and CIFAR100 dataset and dataloader implementations
"""
import numpy as np
import torchvision
import torchvision.transforms as transforms

from datasets.data import MultiTaskDataHandler
from typing import List


class CifarHandler(MultiTaskDataHandler):
    def get_transforms(self,
                       epochs: int):
        # Cannot use entire data statistics
        # Use this mean/std so that no information leaks from entire dataset
        mean_norm = [0.50, 0.50, 0.50]
        std_norm = [0.25, 0.25, 0.25]

        augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_norm, std_norm),
        ])
        vanilla_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)])

        # If training for just 1 epoch, then augmentations are not useful
        if epochs == 1:
            train_transform = vanilla_transform
        else:
            train_transform = augment_transform

        return train_transform, vanilla_transform

    def split_dataset(self,
                      tasks: List[List[int]],
                      replay_frac: int) -> None:
        """
        Use the "tasks" vector to split dataset
        """
        tr_ind, te_ind = [], []
        tr_lab, te_lab = [], []

        for task_id, tsk in enumerate(tasks):
            for lab_id, lab in enumerate(tsk):

                task_tr_ind = np.where(np.isin(self.trainset.targets,
                                               [lab]))[0]
                task_te_ind = np.where(np.isin(self.testset.targets,
                                               [lab]))[0]

                # This only applies if limited replay is to be used:
                #   Reduce replay samples except for the last task which is
                #   allowed to use all available samples. Duplicate older task
                #   samples so that total number of samples of the new task is
                #   close to the number of samples of all other tasks. Every
                #   mini-batch has approximately half the samples from the
                #   new task and half the samples from all other tasks
                if replay_frac < 0.99 and task_id != len(tasks) - 1:
                    samples = int(replay_frac * len(task_tr_ind))
                    copies = 1.0 / ((len(tasks) - 1) * replay_frac)
                    copies = max(int(copies), 1)

                    task_tr_ind = task_tr_ind[:samples]
                    task_tr_ind = np.repeat(task_tr_ind, copies)

                tr_ind.append(task_tr_ind)
                te_ind.append(task_te_ind)

                # Change labels to (task_id, label)
                curlab = (task_id, lab_id)
                tr_vals = [curlab for _ in range(len(task_tr_ind))]
                te_vals = [curlab for _ in range(len(task_te_ind))]

                tr_lab.append(tr_vals)
                te_lab.append(te_vals)

        tr_ind = np.concatenate(tr_ind)
        te_ind = np.concatenate(te_ind)
        self.tr_ind = tr_ind
        self.te_ind = te_ind

        tr_lab, te_lab = np.concatenate(tr_lab), np.concatenate(te_lab)

        self.trainset.data = self.trainset.data[tr_ind]
        self.testset.data = self.testset.data[te_ind]

        self.trainset.targets = [list(it) for it in tr_lab]
        self.testset.targets = [list(it) for it in te_lab]


class Cifar100Handler(CifarHandler):
    """
    Load CIFAR100 and prepare dataset, by splitting it into different
    tasks, based on the config file
    """
    def __init__(self,
                 args,
                 tasks: List[List[int]],
                 limited_replay: bool = False,
                 download: bool = False) -> None:
        """
        Download CIFAR100 and prepare requested config of CIFAR100
        Args:
            - tasks: A List of tasks. Each element in the list is another list
              describing the labels contained in the task. If a 100 is added to
              a task label, then that particular label has randomized labels
            - samples: Number of samples for each label
            - seed: Random seed. A Negative random seed implies that the subset
              of data points chosen is through the np.roll function
        """

        train_transform, test_transform = self.get_transforms(args.epochs)
        self.trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=download,
            transform=train_transform)
        self.testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=download,
            transform=test_transform)

        replay_frac = args.replay_frac if limited_replay else 1
        self.split_dataset(tasks, replay_frac)


class Cifar10Handler(CifarHandler):
    """
    Load CIFAR10 and prepare dataset, by splitting it into different
    tasks, based on the config file
    """
    def __init__(self,
                 args,
                 tasks: List[List[int]],
                 limited_replay: bool = False,
                 download: bool = False) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: A List of tasks. Each element in the list is another list
              describing the labels contained in the task
            - samples: Number of samples for each label
            - seed: Random seed. A Negative random seed implies that the subset
              of data points chosen is through the np.roll function
        """
        train_transform, test_transform = self.get_transforms(args.epochs)

        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=download,
            transform=train_transform)
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=download,
            transform=test_transform)

        replay_frac = args.replay_frac if limited_replay else 1
        self.split_dataset(tasks, replay_frac)
