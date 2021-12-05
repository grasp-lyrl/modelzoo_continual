"""
Modified MNIST dataset to allow for modified target format. If the targets are
changed to have task, information, then the current __getitem__ function fail.
"""
import torchvision

from PIL import Image
from typing import Any, Tuple


class ModMNIST(torchvision.datasets.MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
