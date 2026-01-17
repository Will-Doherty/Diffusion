# Agents

## Project overview
- This repo implements a score-based diffusion/annealed score matching model for MNIST.
- Core code lives in `diffusion/` and is packaged as the `diffusion` module.
- Training and sampling are wired through a single script: `diffusion/annealed_sm_mnist.py`.

## Key files
- `diffusion/annealed_sm_mnist.py`: training loop + inference entrypoint; saves/loads weights and runs annealed Langevin sampling.
- `diffusion/config.py`: dataclass configs for training/inference and weight paths.
- `diffusion/datasets.py`: MNIST dataset wrapper + preprocessing (noise, logit transform, normalization).
- `diffusion/losses.py`: annealed score matching loss (sliced/SM objective).
- `diffusion/models.py`: UNet + RefineNet definitions.
- `diffusion/inference.py`: Langevin and annealed Langevin samplers.
- `diffusion/plotting.py`: plotting stub (currently no implementation).

## Usage
- Train + sample: `python diffusion/annealed_sm_mnist.py`
- Inference only (load existing weights): `python diffusion/annealed_sm_mnist.py --inference_only`
- Weights are saved to `model_weights/sliced_score_matching_MNIST_weights.pt`.
- MNIST data is downloaded to `data/` by default.

## Tests
- `tests/test_sliced_sm_mnist.py` checks UNet output shape. It imports `score_matching_mnist`, which looks outdated vs the current `diffusion` module name.

## Current gaps / TODO
- `diffusion/plotting.py` is empty; sampling results are not visualized yet.
- The training loop in `diffusion/annealed_sm_mnist.py` is minimal (small batch counts, early break) and likely a placeholder.

## Notes
- `diffusion/annealed_sm_mnist.py` forces `matplotlib` to use `TkAgg`, which requires a GUI backend.
