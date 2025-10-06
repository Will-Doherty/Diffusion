import torch
import matplotlib
matplotlib.use("TkAgg")
import argparse

from diffusion.config import SetupConfigGM, TrainingConfigGM, InferenceConfigGM
from diffusion.models import ScoreFnNet
from diffusion.losses import calculate_sm_objective
from diffusion.inference import run_langevin_sampling
from diffusion.plotting import plot_gm_sampling_result

setup_cfg = SetupConfigGM()

def train_gm_score_matching(cfg: TrainingConfigGM):
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

parser = argparse.ArgumentParser()
parser.add_argument("--inference_only", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

training_cfg = TrainingConfigGM()
inference_cfg = InferenceConfigGM()

if __name__ == "__main__":
    if args.inference_only:
        if not setup_cfg.weight_path.exists():
            raise FileNotFoundError("Existing weight path not found. You might need to train the model first!")
        model = ScoreFnNet()
        model.load_state_dict(torch.load(setup_cfg.weight_path))
    else:
        model = train_gm_score_matching(training_cfg)
        setup_cfg.weight_directory.mkdir(exist_ok=True)
        torch.save(model.state_dict(), setup_cfg.weight_path)

    inference_samples = run_langevin_sampling(inference_cfg, model)
    gm = training_cfg.gm
    plot_gm_sampling_result(inference_samples, gm)