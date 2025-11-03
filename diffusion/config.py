from dataclasses import dataclass
from pathlib import Path

from diffusion.datasets import MNIST

# -- MNIST config -- 

@dataclass
class TrainingConfigMNIST:
    lr = 1e-4
    num_epochs = 1
    num_batches = 5
    mnist = MNIST()
    sigmas = [1.0]

@dataclass
class TrainingConfigAnnealedMNIST:
    lr = 1e-4
    num_epochs = 1
    num_batches = 5
    mnist = MNIST()
    sigmas = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]


@dataclass
class SetupConfigMNIST:
    weight_directory = Path("model_weights")
    weight_path = weight_directory / "sliced_score_matching_MNIST_weights.pt"

@dataclass
class InferenceConfigMNIST:
    n_steps = 10_000
    step_size = 0.001
    sigmas = TrainingConfigAnnealedMNIST.sigmas