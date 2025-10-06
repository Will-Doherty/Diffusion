import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class Gaussian2D:
    def __init__(self, mu, sigma):
        self.mu = torch.tensor(mu, dtype=torch.float64)
        self.sigma = torch.tensor(sigma, dtype=torch.float64)
        self.inv_2pi = 1.0 / (2.0 * torch.pi)

    def get_density(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mu) / self.sigma
        exp_term = torch.exp(-0.5 * (z * z).sum(dim=-1))
        norm = self.inv_2pi / (self.sigma.prod())
        return norm * exp_term

    def get_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_score(x) * self.get_density(x)[..., None]

    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        return -(x - self.mu) / (self.sigma * self.sigma)

    def sample(self, n: int) -> torch.Tensor:
        return self.mu + self.sigma * torch.randn(n, 2, dtype=torch.float64)

class GaussianMixture2D:
    def __init__(self, mu1, mu2, sigma1, sigma2, w):
        self.g1 = Gaussian2D(mu1, sigma1)
        self.g2 = Gaussian2D(mu2, sigma2)
        self.w = torch.tensor(w, dtype=torch.float64)

    def get_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * self.g1.get_density(x) + (1.0 - self.w) * self.g2.get_density(x)

    def get_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * self.g1.get_gradient(x) + (1.0 - self.w) * self.g2.get_gradient(x)

    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        p1 = self.w * self.g1.get_density(x)
        p2 = (1.0 - self.w) * self.g2.get_density(x)
        num = p1[..., None] * self.g1.get_score(x) + p2[..., None] * self.g2.get_score(x)
        den = (p1 + p2)[..., None]
        return num / den

    def sample(self, n: int) -> torch.Tensor:
        z = torch.bernoulli(torch.full((n, 1), self.w, dtype=torch.float64))
        x1 = self.g1.sample(n)
        x2 = self.g2.sample(n)
        return z * x1 + (1 - z) * x2


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