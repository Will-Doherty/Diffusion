from dataclasses import dataclass
from pathlib import Path

from diffusion.datasets import GaussianMixture2D
from diffusion.datasets import MNIST

# -- Gaussian mixture config -- 

@dataclass
class SetupConfigGM:
    weight_directory = Path("model_weights")
    weight_path = weight_directory / "sliced_score_matching_gm_weights.pt"

@dataclass
class TrainingConfigGM:
    lr = 1e-5
    n = 30_000
    use_sliced_sm: bool = False
    gm = GaussianMixture2D(mu1=(0.0, 0.0), mu2=(1.0, 1.0), sigma1=(1.0, 1.0), sigma2=(1.0, 1.0), w=0.5)

@dataclass
class InferenceConfigGM:
    n_steps = 10_000
    initial_x = (0, 0)
    step_size = 0.001

# -- MNIST config -- 

@dataclass
class TrainingConfigMNIST:
    lr = 1e-4
    num_epochs = 1
    num_batches = 5
    use_sliced_sm: bool = False
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