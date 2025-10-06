from diffusion.sliced_score_matching_gm import ScoreFnNet
from diffusion.config import SetupConfigGM, TrainingConfigGM

import torch

setup_cfg = SetupConfigGM()
training_cfg = TrainingConfigGM()
 
def compare_gradients():
    if not setup_cfg.weight_path.exists():
        raise FileNotFoundError("Existing weight path not found. You might need to train the model first!")
    model = ScoreFnNet()
    model.load_state_dict(torch.load(setup_cfg.weight_path))
    gm = training_cfg.gm
    input_samples = [gm.sample(1)[0].to(torch.float32) for _ in range(10)]
    scores = []
    predictions = []
    for x in input_samples:
        x = x.clone().requires_grad_(True)
        energy = model(x).sum()
        grad_x = -torch.autograd.grad(energy, x)[0]
        scores.append(gm.get_score(x.detach()))
        predictions.append(grad_x.detach())
    return scores, predictions

if __name__ == "__main__":
    scores, predictions = compare_gradients()
    print(f"{'idx':>3} | {'score':>24} | {'pred':>24}")
    print("-"*3 + " | " + "-"*24 + " | " + "-"*24)
    for i, (s, p) in enumerate(zip(scores, predictions)):
        s = s.detach().view(-1).tolist()
        p = p.detach().view(-1).tolist()
        s_str = " ".join(f"{v:>10.6f}" for v in s)
        p_str = " ".join(f"{v:>10.6f}" for v in p)
        print(f"{i:>3} | {s_str:<24} | {p_str:<24}")




