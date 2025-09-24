import torch
import torch.nn as nn
from torch.nn.functional import relu, mse_loss
from dataclasses import dataclass
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import subprocess
from tqdm import tqdm


# --- DATASET GENERATION ---

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

    def sample(self, n: int) -> torch.Tensor:
        return self.mu + self.sigma * torch.randn(n, dtype=torch.float64)

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

    def sample(self, n: int) -> torch.Tensor:
        z = torch.bernoulli(torch.full((n,), self.w, dtype=torch.float64))
        x1 = self.g1.sample(n)
        x2 = self.g2.sample(n)
        return z * x1 + (1 - z) * x2


# --- TRAINING CODE ---

class ScoreFnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1, bias=False)

        nn.init.normal_(self.layer1.weight, std=0.02)
        nn.init.normal_(self.layer2.weight, std=0.02)
        nn.init.normal_(self.layer3.weight, std=0.02)

    def forward(self, x):
        # TODO - condition on time
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = self.layer3(x)
        return x

@dataclass
class Config:
    lr = 1e-4
    n = 30_000

def train_score_matching():
    cfg = Config()
    model = ScoreFnNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gm = GaussianMixture(0.0, 3.0, 1.0, 1.0, 0.5)
    losses = []
    print("Training...")
    for i in tqdm(range(cfg.n)):
        x = gm.sample(n=1).to(torch.float32)
        target = gm.get_score(x).to(torch.float32)
        model.zero_grad(set_to_none=True)
        output = model(x)
        loss = mse_loss(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return model


# --- INFERENCE CODE ---

def langevin_step(current_x: torch.Tensor, step_size: float, gm: GaussianMixture, model) -> torch.Tensor:
    noise = torch.randn((), dtype=torch.float64)
    return current_x + 0.5 * step_size * model(current_x) + torch.sqrt(torch.tensor(step_size, dtype=torch.float64)) * noise

def run_langevin_sampling(n_steps: int, initial_x: float, step_size: float, gm: GaussianMixture) -> torch.Tensor:
    samples = torch.zeros(n_steps, dtype=torch.float64)
    current_x = torch.tensor(initial_x, dtype=torch.float64)
    samples[0] = current_x
    for i in range(1, n_steps):
        current_x = langevin_step(current_x, step_size, gm)
        samples[i] = current_x
    return samples


if __name__ == "__main__":
    trained_model = train_score_matching()
    run_langevin_sampling()

