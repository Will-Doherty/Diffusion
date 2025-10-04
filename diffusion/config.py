from dataclasses import dataclass
from pathlib import Path

from diffusion.gaussian_mixtures import GaussianMixture2D

@dataclass
class SetupConfig:
    weight_directory = Path("model_weights")
    weight_path = weight_directory / "sliced_score_matching_gm_weights.pt"

@dataclass
class TrainingConfig:
    lr = 1e-5
    n = 30_000
    use_sliced_sm: bool = False
    gm = GaussianMixture2D(mu1=(0.0, 0.0), mu2=(1.0, 1.0), sigma1=(1.0, 1.0), sigma2=(1.0, 1.0), w=0.5)

@dataclass
class InferenceConfig:
    n_steps = 10_000
    initial_x = (0, 0)
    step_size = 0.001

