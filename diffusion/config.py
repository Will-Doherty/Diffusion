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

@dataclass
class SetupConfigMNIST:
    weight_directory = Path("model_weights")
    weight_path = weight_directory / "sliced_score_matching_MNIST_weights.pt"

@dataclass
class InferenceConfigMNIST:
    n_steps = 10_000
    # initial_x = 
    step_size = 0.001