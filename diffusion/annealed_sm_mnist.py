import torch
import matplotlib
matplotlib.use("TkAgg")
import argparse

from diffusion.config import TrainingConfigAnnealedMNIST, InferenceConfigMNIST, SetupConfigMNIST
from diffusion.models import UNet
from diffusion.losses import calculate_annealed_sm_objective_mnist
from diffusion.inference import run_annealed_langevin_sampling
from diffusion.plotting import plot_mnist_sampling_result

setup_cfg = SetupConfigMNIST()

def train_annealed_mnist_score_matching(cfg: TrainingConfigAnnealedMNIST):
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    dset = cfg.mnist
    loader = dset.train_loader
    losses = []
    print("Training...")
    for sigma in cfg.sigmas:
        for i, (x, _) in enumerate(loader):
            model.zero_grad(set_to_none=True)
            loss = calculate_annealed_sm_objective_mnist(model, x, sigma)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            print(f"step {i}, loss: {sum(losses[-100:]) / 100}")
            if i % 100 == 0 and i > 0:
                break
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--inference_only", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

training_cfg = TrainingConfigAnnealedMNIST()
inference_cfg = InferenceConfigMNIST()

if __name__ == "__main__":
    if args.inference_only:
        if not setup_cfg.weight_path.exists():
            raise FileNotFoundError("Existing weight path not found. You might need to train the model first!")
        model = UNet()
        model.load_state_dict(torch.load(setup_cfg.weight_path))
    else:
        model = train_annealed_mnist_score_matching(training_cfg)
        setup_cfg.weight_directory.mkdir(exist_ok=True)
        torch.save(model.state_dict(), setup_cfg.weight_path)

    inference_samples = run_annealed_langevin_sampling(inference_cfg, model)
    plot_mnist_sampling_result(inference_samples)