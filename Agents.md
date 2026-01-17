# Agents

## Project overview
- This repo implements a score-based diffusion model for MNIST using annealed score matching and a small UNet.
- Core code lives in `diffusion/` and is packaged as the `diffusion` module.
- Training and sampling are wired through a single script: `diffusion/annealed_sm_mnist.py`.

## Key files
- `diffusion/annealed_sm_mnist.py`: training loop + inference entrypoint; saves/loads weights and runs annealed Langevin sampling.
- `diffusion/config.py`: dataclass configs for training/inference, sigma schedule, and weight/sample paths.
- `diffusion/datasets.py`: MNIST dataset wrapper + preprocessing (tensor + normalization).
- `diffusion/losses.py`: annealed score matching objective.
- `diffusion/models.py`: UNet backbone with sigma conditioning.
- `diffusion/inference.py`: annealed Langevin sampling + saving MNIST samples/grids.
- `colab_train.ipynb`: notebook entrypoint for training in Colab.

## Usage
- Train + sample: `python diffusion/annealed_sm_mnist.py`
- Inference only (load existing weights): `python diffusion/annealed_sm_mnist.py --inference_only`
- Weights are saved to `model_weights/sliced_score_matching_MNIST_weights.pt`.
- Samples are written to `model_samples/` (`grid.png` plus per-sample images).
- MNIST data is downloaded to `data/` by default.

## Tests
- `tests/test_sliced_sm_mnist.py` checks UNet output shape, but it imports `score_matching_mnist` instead of the `diffusion` package.

## Current gaps / TODO
- No standalone plotting/visualization module; sampling only writes images via `diffusion/inference.py`.
- The training loop is intentionally small (5 epochs, early stopping patience=1) and may be a placeholder for longer runs.

## Notes
- `pyproject.toml` targets Python 3.12 and declares CPU-only PyTorch indices via `uv`.
