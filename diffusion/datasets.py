from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch

class MNIST:
    def __init__(
        self,
        root: str = "data",
        batch_size: int = 2,
        num_workers: int = 2,
        download: bool = True,
        shuffle_train: bool = True,
        shuffle_test: bool = False,
    ):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.train_dataset = datasets.MNIST(
            root=root, train=True, download=download, transform=transform
        )
        self.test_dataset = datasets.MNIST(
            root=root, train=False, download=download, transform=transform
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=num_workers,
            pin_memory=True,
        )
