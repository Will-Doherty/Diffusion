import torch
import torch.nn as nn
from torch.nn.functional import gelu, mse_loss
from torch.func import hessian
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from diffusion.config import SetupConfig, TrainingConfig, InferenceConfig
from diffusion.models import ScoreFnNet

setup_cfg = SetupConfig()


# --- TRAINING CODE ---

def calculate_sm_objective(model, x):
    # TODO: add batching
    x = x.squeeze(0).requires_grad_()
    energy = model(x)
    score = -torch.autograd.grad(energy, x, create_graph=True)[0]
    loss1 = 0.5 * (score ** 2).sum(dim=-1)
    loss2 = 0.0
    D = x.shape[-1]
    for i in range(D):
        gi = torch.autograd.grad(score[..., i].sum(), x, create_graph=True, retain_graph=True)[0][..., i]
        loss2 = loss2 + gi
    return (loss1 + loss2).mean()

# def calculate_sliced_sm_objective(model, x):
#     x = x.squeeze(0).requires_grad_(True)
#     v = torch.randn_like(x)

#     y = model(x).sum()
#     score = torch.autograd.grad(y, x, create_graph=True)[0]
#     Hv = torch.autograd.grad((score * v).sum(), x, create_graph=True)[0]

#     first_summand = (v * Hv).sum()
#     second_summand = 0.5 * ((v * score).sum())**2
#     return first_summand + second_summand

def train_score_matching(cfg: TrainingConfig):
    model = ScoreFnNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gm = cfg.gm
    losses = []
    print("Training...")
    for i in range(cfg.n):
        x = gm.sample(n=1).to(torch.float32)
        model.zero_grad(set_to_none=True)
        if cfg.use_sliced_sm:
            continue  # temporary
            # loss = calculate_sliced_sm_objective(model, x)
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
    current_x.requires_grad_()
    score = -torch.autograd.grad(model(current_x).sum(), current_x)[0]
    noise = torch.randn_like(current_x)
    step = current_x + 0.5 * step_size * score + torch.sqrt(torch.tensor(step_size, dtype=torch.float32)) * noise
    return step.detach()

def run_langevin_sampling(cfg: InferenceConfig, model) -> torch.Tensor:
    n_steps, initial_x, step_size = cfg.n_steps, cfg.initial_x, cfg.step_size
    samples = torch.zeros((n_steps, 2), dtype=torch.float32)
    current_x = torch.tensor(initial_x, dtype=torch.float32)
    samples[0] = current_x
    for i in tqdm(range(1, n_steps), desc="Running Langevin Sampling"):
        current_x = langevin_step(current_x, step_size, model)
        samples[i] = current_x
    return samples

parser = argparse.ArgumentParser()
parser.add_argument("--inference_only", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

training_cfg = TrainingConfig()
inference_cfg = InferenceConfig()

if __name__ == "__main__":
    if args.inference_only:
        if not setup_cfg.weight_path.exists():
            raise FileNotFoundError("Existing weight path not found. You might need to train the model first!")
        model = ScoreFnNet()
        model.load_state_dict(torch.load(setup_cfg.weight_path))
    else:
        model = train_score_matching(training_cfg)
        setup_cfg.weight_directory.mkdir(exist_ok=True)
        torch.save(model.state_dict(), setup_cfg.weight_path)

    inference_samples = run_langevin_sampling(inference_cfg, model)
    gm = training_cfg.gm
    
    # --- Plotting ---
    x_range = torch.linspace(-5, 8, 100, dtype=torch.float64)
    y_range = torch.linspace(-5, 8, 100, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    
    density = gm.get_density(grid)
    
    plt.figure(figsize=(8, 8))
    plt.contourf(grid_x.numpy(), grid_y.numpy(), density.numpy(), levels=20, cmap='viridis')
    plt.colorbar(label='Density')
    
    samples_np = inference_samples.detach().numpy()
    plt.scatter(samples_np[:, 0], samples_np[:, 1], c='r', label='Langevin Samples', marker='x', s=10, alpha=0.5)
    
    plt.title('True Distribution and Langevin Samples')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
