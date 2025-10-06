import torch
from tqdm import tqdm

# -- STANDARD LANGEVIN SAMPLING --

def langevin_step(current_x: torch.Tensor, step_size: float, model) -> torch.Tensor:
    current_x.requires_grad_()
    score = -torch.autograd.grad(model(current_x).sum(), current_x)[0]
    noise = torch.randn_like(current_x)
    step = current_x + 0.5 * step_size * score + torch.sqrt(torch.tensor(step_size, dtype=torch.float32)) * noise
    return step.detach()

def run_langevin_sampling(cfg, model) -> torch.Tensor:
    n_steps, initial_x, step_size = cfg.n_steps, cfg.initial_x, cfg.step_size
    samples = torch.zeros((n_steps, 2), dtype=torch.float32)
    current_x = torch.tensor(initial_x, dtype=torch.float32)
    samples[0] = current_x
    for i in tqdm(range(1, n_steps), desc="Running Langevin Sampling"):
        current_x = langevin_step(current_x, step_size, model)
        samples[i] = current_x
    return samples


# -- ANNEALED LANGEVIN SAMPLING --

def annealed_langevin_step(current_x: torch.Tensor, step_size: float, model) -> torch.Tensor:
    # TODO - parameterize this function and the one below with time
    noise = torch.randn((), dtype=torch.float64)
    return current_x + 0.5 * step_size * model(current_x) + torch.sqrt(torch.tensor(step_size, dtype=torch.float64)) * noise

def run_annealed_langevin_sampling(cfg, model) -> torch.Tensor:
    n_steps, initial_x, step_size = cfg.n_steps, cfg.initial_x, cfg.step_size
    samples = torch.zeros(n_steps, dtype=torch.float64)
    current_x = torch.tensor(initial_x, dtype=torch.float64)
    samples[0] = current_x
    for i in range(1, n_steps):
        current_x = annealed_langevin_step(current_x, step_size, model)
        samples[i] = current_x
    return samples