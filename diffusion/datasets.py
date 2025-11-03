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
        transform: transforms.Compose | None = None,
    ):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(self.add_uniform_noise),
                transforms.Lambda(self.logit_transform),
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

    def add_uniform_noise(self, img, noise_range=(-1/512, 1/512), clip_range=(-0.001, 0.001)):
        noise = torch.rand_like(img) * (noise_range[1] - noise_range[0]) + noise_range[0]
        noise = torch.clamp(noise, clip_range[0], clip_range[1])
        return img + noise

    def logit_transform(self, img, alpha=1e-6):
        img = img * (1 - 2 * alpha) + alpha
        return torch.log(img / (1 - img))