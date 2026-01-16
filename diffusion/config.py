from dataclasses import dataclass
from pathlib import Path

from diffusion.datasets import MNIST

def geometric_sigmas(num_levels: int = 20, sigma_max: float = 1.0, sigma_min: float = 0.01) -> list[float]:
    ratio = sigma_min / sigma_max
    return [sigma_max * (ratio ** (i / (num_levels - 1))) for i in range(num_levels)]

@dataclass
class TrainingConfigAnnealedMNIST:
    lr = 1e-6
    num_epochs = 1
    num_batches = 5
    mnist = MNIST()
    sigmas = geometric_sigmas()

@dataclass
class SetupConfigMNIST:
    weight_directory = Path("model_weights")
    weight_path = weight_directory / "sliced_score_matching_MNIST_weights.pt"

@dataclass
class InferenceConfigMNIST:
    n_steps = 10_000
    step_size = 0.001
    sigmas = TrainingConfigAnnealedMNIST.sigmas
