"""
Dataset loader for CIFAR100
"""
from copy import deepcopy

import torch
import numpy as np

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def wif(id):
    """
    Used to fix randomization bug for pytorch dataloader + numpy
    Code from https://github.com/pytorch/pytorch/issues/5059
    """
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


class MultiTaskDataHandler():
    """
    Template class for a Multi-task data handler
    """
    def __init__(self) -> None:
        self.trainset: Dataset
        self.testset: Dataset

    def get_data_loader(self,
                        batch_size: int,
                        workers: int,
                        train: bool = True) -> DataLoader:
        """
        Get the Dataloader for the entire dataset
        Args:
            - shuf       : Shuffle
            - wtd_loss   : Dataloader also has wts along with targets
            - wtd_sampler: Sample data from dataloader with weights
                         according to self.tr_wts
        """
        data = self.trainset if train else self.testset
        loader = DataLoader(
            data, batch_size=batch_size, shuffle=train,
            num_workers=workers, pin_memory=True,
            worker_init_fn=wif)

        return loader

    def get_task_data_loader(self,
                             task: int,
                             batch_size: int,
                             workers: int,
                             train: bool = False) -> DataLoader:
        """
        Get Dataloader for a specific task
        """
        if train:
            task_set = deepcopy(self.trainset)
        else:
            task_set = deepcopy(self.testset)

        task_ind = [task == i[0] for i in task_set.targets]

        task_set.data = task_set.data[task_ind]
        task_set.targets = np.array(task_set.targets)[task_ind, :]
        task_set.targets = [(lab[0], lab[1]) for lab in task_set.targets]

        loader = DataLoader(
            task_set, batch_size=batch_size,
            shuffle=False, num_workers=workers, pin_memory=True,
            worker_init_fn=wif)

        return loader
