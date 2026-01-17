from pathlib import Path
import torch
from torchvision.utils import save_image
from tqdm import tqdm


def annealed_langevin_step(current_x: torch.Tensor, step_size: float, sigma: float, model) -> torch.Tensor:
    current_x.requires_grad_()
    score = -torch.autograd.grad(model(current_x, sigma).sum(), current_x)[0]
    noise = torch.randn_like(current_x)
    step_size_tensor = torch.tensor(step_size, dtype=current_x.dtype, device=current_x.device)
    step = current_x + 0.5 * step_size * score + torch.sqrt(step_size_tensor) * noise
    return step.detach()


def run_annealed_langevin_sampling(cfg, model) -> torch.Tensor:
    n_steps, sigmas, step_size = cfg.n_steps, cfg.sigmas, cfg.step_size
    device = next(model.parameters()).device
    initial_x = torch.randn(cfg.n_samples, 1, 28, 28, device=device)
    current_x = initial_x
    for sigma in tqdm(sigmas, desc="Running Annealed Langevin Sampling"):
        for i in range(n_steps):
            current_x = annealed_langevin_step(current_x, step_size, sigma, model)
    return current_x


def save_mnist_samples_to_dir(
    samples: torch.Tensor,
    output_dir: str | Path,
    mean: float = 0.1307,
    std: float = 0.3081,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "grid.png"
    with torch.no_grad():
        x = samples.detach().cpu()
        x = x * std + mean
        x = torch.clamp(x, 0.0, 1.0)
        save_image(x, grid_path, nrow=int(len(x) ** 0.5) or 1)
        for idx, sample in enumerate(x):
            save_image(sample, output_dir / f"sample_{idx:03d}.png")
    return output_dir
