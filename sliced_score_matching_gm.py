import torch
import torch.nn as nn
from torch.nn.functional import gelu, mse_loss
from torch.func import hessian
from dataclasses import dataclass
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path


# --- DATASET GENERATION ---

class Gaussian:
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
        num = p1[..., None] * self.g1.get_score(x) + p2[..., None] * self.g2.get_score(x)
        den = (p1 + p2)[..., None]
        return num / den

    def sample(self, n: int) -> torch.Tensor:
        z = torch.bernoulli(torch.full((n, 1), self.w, dtype=torch.float64))
        x1 = self.g1.sample(n)
        x2 = self.g2.sample(n)
        return z * x1 + (1 - z) * x2


# --- TRAINING CODE ---

def calculate_sm_objective(model, x):
    # TODO: add batching
    x = x.squeeze(0)
    def scalar_fn(inp):
        return model(inp).sum()
    loss = (
        torch.trace(hessian(scalar_fn)(x))
        + 0.5 * torch.linalg.vector_norm(torch.cat([p.reshape(-1) for p in model.parameters()]))
    )
    return loss

def calculate_sliced_sm_objective(model, x):
    # TODO: add batching
    x = x.squeeze(0).requires_grad_()

    def scalar_fn(inp):
        return model(inp).sum()
    
    v = torch.randn_like(x)
    Hv = torch.autograd.functional.hvp(scalar_fn, x, v)[1]
    first_summand = v @ Hv
    second_summand = 0.5 * (v @ torch.autograd.grad(model(x), x)[0]).pow(2)
    loss = first_summand + second_summand
    return loss

class ScoreFnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1, bias=False)

        nn.init.normal_(self.layer1.weight, std=0.02)
        nn.init.normal_(self.layer2.weight, std=0.02)
        nn.init.normal_(self.layer3.weight, std=0.02)

    def forward(self, x):
        # TODO - condition on time
        x = gelu(self.layer1(x))  # swapped to gelu because relu has 0s everywhere in its Hessian
        x = gelu(self.layer2(x))
        x = self.layer3(x)
        return x

@dataclass
class TrainingConfig:
    lr = 1e-4
    n = 100
    use_sliced_sm: bool = True

@dataclass
class InferenceConfig:
    n_steps = 10
    initial_x = (0, 0)
    step_size = 0.0001

def train_score_matching(cfg: TrainingConfig):
    model = ScoreFnNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gm = GaussianMixture((0.0, 0.0), (3.0, 3.0), (1.0, 1.0), (1.0, 1.0), 0.5)
    losses = []
    print("Training...")
    for i in range(cfg.n):
        x = gm.sample(n=1).to(torch.float32)
        model.zero_grad(set_to_none=True)
        if cfg.use_sliced_sm:
            loss = calculate_sliced_sm_objective(model, x)
            break
        else:
            loss = calculate_sm_objective(model, x)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if i % 100 == 0 and i > 0:
            print(f"step {i}, loss: {sum(losses[-100:]) / 100:.6f}")
    return model


# --- INFERENCE CODE ---

def langevin_step(current_x: torch.Tensor, step_size: float, model) -> torch.Tensor:
    noise = torch.randn((), dtype=torch.float32)
    return current_x + 0.5 * step_size * model(current_x) + torch.sqrt(torch.tensor(step_size, dtype=torch.float64)) * noise

def run_langevin_sampling(cfg: InferenceConfig, model) -> torch.Tensor:
    n_steps, initial_x, step_size = cfg.n_steps, cfg.initial_x, cfg.step_size
    samples = torch.zeros((n_steps, 2), dtype=torch.float32)
    current_x = torch.tensor(initial_x, dtype=torch.float32)
    samples[0] = current_x
    for i in range(1, n_steps):
        current_x = langevin_step(current_x, step_size, model)
        samples[i] = current_x
    return samples


if __name__ == "__main__":
    weight_directory = Path("model_weights")
    weight_path = weight_directory / "sliced_score_matching_gm_weights.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_only", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.inference_only:
        if not weight_path.exists():
            raise FileNotFoundError("Existing weight path not found. You might need to train the model first!")
        model = ScoreFnNet().load_state_dict(weight_path) 
    else:
        training_cfg = TrainingConfig()
        model = train_score_matching(training_cfg)
        weight_directory.mkdir(exist_ok=True)
        torch.save(model.state_dict(), weight_path)

    inference_cfg = InferenceConfig()
    inference_samples = run_langevin_sampling(inference_cfg, model)
