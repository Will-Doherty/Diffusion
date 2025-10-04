import torch


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = torch.tensor(mu, dtype=torch.float64)
        self.sigma = torch.tensor(sigma, dtype=torch.float64)
        self.inv_sqrt_2pi = 1.0 / torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=torch.float64))

    def get_density(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mu) / self.sigma
        return self.inv_sqrt_2pi / self.sigma * torch.exp(-0.5 * z * z)

    def get_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return -(x - self.mu) / (self.sigma * self.sigma) * self.get_density(x)

    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        return -(x - self.mu) / (self.sigma * self.sigma)


class GaussianMixture:
    def __init__(self, mu1, mu2, sigma1, sigma2, w):
        self.g1 = Gaussian(mu1, sigma1)
        self.g2 = Gaussian(mu2, sigma2)
        self.w = torch.tensor(w, dtype=torch.float64)

    def get_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * self.g1.get_density(x) + (1.0 - self.w) * self.g2.get_density(x)

    def get_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * self.g1.get_gradient(x) + (1.0 - self.w) * self.g2.get_gradient(x)

    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        p1 = self.w * self.g1.get_density(x)
        p2 = (1.0 - self.w) * self.g2.get_density(x)
        numerator = p1 * self.g1.get_score(x) + p2 * self.g2.get_score(x)
        denominator = p1 + p2
        return numerator / denominator


def langevin_step(current_x: torch.Tensor, step_size: float, gm: GaussianMixture) -> torch.Tensor:
    noise = torch.randn((), dtype=torch.float64)
    return current_x + 0.5 * step_size * gm.get_score(current_x) + torch.sqrt(torch.tensor(step_size, dtype=torch.float64)) * noise


def run_langevin_sampling(n_steps: int, initial_x: float, step_size: float, gm: GaussianMixture) -> torch.Tensor:
    samples = torch.zeros(n_steps, dtype=torch.float64)
    current_x = torch.tensor(initial_x, dtype=torch.float64)
    samples[0] = current_x
    for i in range(1, n_steps):
        current_x = langevin_step(current_x, step_size, gm)
        samples[i] = current_x
    return samples


if __name__ == "__main__":
    gm = GaussianMixture(0.0, 3.0, 1.0, 1.0, 0.5)
    samples = run_langevin_sampling(100000, 0.5, 0.01, gm)
    torch.save(samples, "outputs/samples.pt")