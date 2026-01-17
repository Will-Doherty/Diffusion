from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch

class MNIST:
    def __init__(
        self,
        root: str = "data",
        batch_size: int = 512,
        num_workers: int = 2,
        download: bool = True,
        shuffle_train: bool = True,
        shuffle_test: bool = False,
        val_split: float = 0.1,
        seed: int = 0,
    ):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        full_train_dataset = datasets.MNIST(
            root=root, train=True, download=download, transform=transform
        )
        self.test_dataset = datasets.MNIST(
            root=root, train=False, download=download, transform=transform
        )

        if not 0.0 < val_split < 1.0:
            raise ValueError("val_split must be between 0 and 1")
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size], generator=generator
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
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
