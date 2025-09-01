import torch
from torchvision.datasets import MNIST
import torch.nn as nn


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