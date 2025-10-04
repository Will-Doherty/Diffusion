import torch

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


