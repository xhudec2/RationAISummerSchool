# Copyright (c) The RationAI team.

import lightning as L
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage: str) -> None:
        self.mnist_train = MNIST(
            self.data_dir, train=True, download=True, transform=self.transform
        )
        self.mnist_val = MNIST(
            self.data_dir, train=False, download=True, transform=self.transform
        )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
