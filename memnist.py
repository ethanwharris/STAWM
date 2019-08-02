import random

import torch
from torchvision.datasets import MNIST


class MemNIST(MNIST):
    def __init__(self, root, number=4, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, target_transform=target_transform, download=download)

        self._transform = transform
        self.number = number

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        imgs = [self._transform(img)]
        targets = [target]

        for i in range(self.number - 1):
            img, target = super().__getitem__(random.randint(0, len(self) - 1))
            imgs.append(self._transform(img))
            targets.append(target)

        img = torch.cat(imgs, 0)
        target = torch.tensor(targets)

        return img, target
