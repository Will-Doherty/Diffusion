import copy
import random
from pathlib import Path
import torch
import argparse

from diffusion.config import TrainingConfigAnnealedMNIST, InferenceConfigMNIST, SetupConfigMNIST
from diffusion.models import UNet
from diffusion.losses import calculate_annealed_sm_objective_mnist
from diffusion.inference import run_annealed_langevin_sampling, save_mnist_samples_to_dir


setup_cfg = SetupConfigMNIST()

def train_annealed_mnist_score_matching(cfg: TrainingConfigAnnealedMNIST):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    dset = cfg.mnist
    loader = dset.train_loader
    val_loader = dset.val_loader
    losses = []
    best_val = float("inf")
    best_state = None
    patience_left = cfg.early_stopping_patience
    print("Training...")
    for i in range(cfg.num_epochs):
        for j, (x, _) in enumerate(loader):
            x = x.to(device)
            sigma = random.choice(cfg.sigmas)
            model.zero_grad(set_to_none=True)
            loss = calculate_annealed_sm_objective_mnist(model, x, sigma)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            window = min(10, len(losses))
            if j % 10 == 0 and j > 0:
                print(f"epoch {i}, step {j}, loss: {sum(losses[-window:]) / window}")

        model.eval()
        val_losses = []
        for x, _ in val_loader:
            x = x.to(device)
            sigma = random.choice(cfg.sigmas)
            val_loss = calculate_annealed_sm_objective_mnist(model, x, sigma)
            val_losses.append(val_loss.item())
        avg_val = sum(val_losses) / max(1, len(val_losses))
        print(f"epoch {i}, val_loss: {avg_val}")

        if avg_val < best_val - cfg.early_stopping_min_delta:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break
        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--inference_only", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

training_cfg = TrainingConfigAnnealedMNIST()
inference_cfg = InferenceConfigMNIST()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.inference_only:
        if not setup_cfg.weight_path.exists():
            raise FileNotFoundError("Existing weight path not found. You might need to train the model first!")
        model = UNet().to(device)
        model.load_state_dict(torch.load(setup_cfg.weight_path, map_location=device))
    else:
        model = train_annealed_mnist_score_matching(training_cfg)
        setup_cfg.weight_directory.mkdir(exist_ok=True)
        torch.save(model.state_dict(), setup_cfg.weight_path)

    inference_samples = run_annealed_langevin_sampling(inference_cfg, model)
    save_mnist_samples_to_dir(inference_samples, setup_cfg.sample_directory)
