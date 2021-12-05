from typing import List

import numpy as np
import torchvision.transforms as transforms

from numpy.random import default_rng
from datasets.modmnist import ModMNIST
from datasets.data import MultiTaskDataHandler


class MNISTHandler(MultiTaskDataHandler):
    def get_transforms(self,
                       epochs: int):
        mean_norm = [0.50]
        std_norm = [0.25]

        augment_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean_norm, std_norm),
        ])
        vanilla_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)])

        if epochs == 1:
            train_transform = vanilla_transform
        else:
            train_transform = augment_transform

        return train_transform, vanilla_transform

    def split_dataset(self,
                      tasks: List[List[int]],
                      replay_frac: int) -> None:
        tr_ind, te_ind = [], []
        tr_lab, te_lab = [], []
        for task_id, tsk in enumerate(tasks):
            for lab_id, lab in enumerate(tsk):

                task_tr_ind = np.where(np.isin(self.trainset.targets,
                                               [lab % 10]))[0]
                task_te_ind = np.where(np.isin(self.testset.targets,
                                               [lab % 10]))[0]

                if replay_frac < 0.99 and task_id != len(tasks) - 1:
                    samples = int(replay_frac * len(task_tr_ind))
                    copies = 1.0 / ((len(tasks) - 1) * replay_frac)
                    copies = max(int(copies), 1)

                    task_tr_ind = task_tr_ind[:samples]
                    task_tr_ind = np.repeat(task_tr_ind, copies)

                tr_ind.append(task_tr_ind)
                te_ind.append(task_te_ind)
                curlab = (task_id, lab_id)

                tr_vals = [curlab for _ in range(len(task_tr_ind))]
                te_vals = [curlab for _ in range(len(task_te_ind))]

                tr_lab.append(tr_vals)
                te_lab.append(te_vals)

        tr_ind, te_ind = np.concatenate(tr_ind), np.concatenate(te_ind)
        tr_lab, te_lab = np.concatenate(tr_lab), np.concatenate(te_lab)

        self.trainset.data = self.trainset.data[tr_ind]
        self.testset.data = self.testset.data[te_ind]

        self.trainset.targets = [list(it) for it in tr_lab]
        self.testset.targets = [list(it) for it in te_lab]

        return tr_lab, te_lab


class SplitMNISTHandler(MNISTHandler):
    """
    Load SplitMNIST dataset Split 10 classes into multiple tasks
    """
    def __init__(self,
                 args,
                 tasks: List[List[int]]) -> None:
        """
        Download dataset and define transforms
        Args:
            - args: Argparse arguments
            - tasks: List of lists. Each inner list is a description of the
        """

        train_transform, test_transform = self.get_trasforms(args.epochs)
        self.trainset = ModMNIST(
            root='./data', train=True, download=True,
            transform=train_transform)
        self.testset = ModMNIST(
            root='./data', train=False, download=True,
            transform=test_transform)

        self.split_dataset(tasks, args.replay_frac)


class RotatedMNISTHandler(MultiTaskDataHandler):
    """
    Rotated MNIST dataset
    """
    def __init__(self,
                 args,
                 tasks: List[List[int]]) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: List of lists. Each inner list is a description of the
              labels that describe a task. If the labe, is 10 * x + y, then the
              label y is rotated by an angle of 10 * x (rotataions are only
              multiples of 10)
            - samples: Number of samples for each label
            - seed: Random seed
        """

        train_transform, test_transform = self.get_trasforms(args.epochs)
        self.trainset = ModMNIST(
            root='./data', train=True, download=True,
            transform=train_transform)
        self.testset = ModMNIST(
            root='./data', train=False, download=True,
            transform=test_transform)

        tr_lab, te_lab = self.split_dataset(tasks, args.replay_frac)

        # Rotate images for each of the tasks based on task_id
        for t_id in range(len(tasks)):
            ang = (tasks[t_id][0] // 10) * 10

            task_tr_flag = tr_lab[:, 0] == t_id
            task_te_flag = te_lab[:, 0] == t_id

            self.trainset.data[task_tr_flag] = transforms.functional.rotate(
                self.trainset.data[task_tr_flag], angle=ang)
            self.testset.data[task_te_flag] = transforms.functional.rotate(
                self.testset.data[task_te_flag], angle=ang)


class PermutedMNISTHandler(MultiTaskDataHandler):
    """
    Initialization for Permuted MNIST dataset
    """
    def __init__(self,
                 args,
                 tasks: List[List[int]]) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: List of lists. Each inner list is a description of the
              labels that describe a task. A label of 10*x + y, then the digit y
              is permuted using random seed (1000*x)
            - samples: Number of samples for each label
            - seed: Random seed
        """
        train_transform, test_transform = self.get_trasforms(args.epochs)
        self.trainset = ModMNIST(
            root='./data', train=True, download=True,
            transform=train_transform)
        self.testset = ModMNIST(
            root='./data', train=False, download=True,
            transform=test_transform)

        tr_lab, te_lab = self.split_dataset(tasks, args.replay_frac)

        # Permute images for each of the tasks based on task_id
        for t_id in range(len(tasks)):
            task_tr_flag = tr_lab[:, 0] == t_id
            task_te_flag = te_lab[:, 0] == t_id

            # Set seed based on task descriptors
            tseed = (tasks[t_id][0] // 10) * 1000
            rng_permute = default_rng(seed=tseed)
            if (tseed == 0):
                idx_permute = np.arange(784)
            else:
                idx_permute = rng_permute.permutation(784)

            self.trainset.data[task_tr_flag] = self.trainset.data[
                task_tr_flag].view(-1, 784)[:, idx_permute].view(-1, 28, 28)
            self.testset.data[task_te_flag] = self.testset.data[
                task_te_flag].view(-1, 784)[:, idx_permute].view(-1, 28, 28)
