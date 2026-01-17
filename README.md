# Score Based Modelling (MNIST)

This repo implements a score-based diffusion model for MNIST, using annealed score matching with a small UNet backbone. Training, sampling, and configuration live in the `diffusion/` package.

## Overview

Run `python diffusion/annealed_sm_mnist.py` to train the UNet score model. Training uses a train/validation split of MNIST with early stopping (configurable in `diffusion/config.py`) and saves weights to `model_weights/`. You can adjust batch size, sigma schedule, and early-stopping settings via `TrainingConfigAnnealedMNIST`.

Run `python diffusion/annealed_sm_mnist.py --inference_only` to load saved weights and generate samples. Outputs are written to `model_samples/` by default.

## Details

### Score matching objective
Score matching trains the network to approximate the gradient of the log density of noisy data. This gives a vector field that points toward higher-probability regions, which can be used during sampling.

### Annealed noise schedule
Annealing introduces multiple noise levels so the model learns scores for progressively less noisy versions of the data. Sampling then follows this same schedule, starting from high noise and moving toward the data distribution.

### Langevin dynamics overview
Langevin sampling iteratively updates a sample by moving in the direction of the model's score (the gradient of log density) and adding Gaussian noise at each step. This alternates between following the learned gradient field and injecting randomness.

### Sigma transitions during sampling
The score model is trained to estimate the score of data corrupted at a specific noise level. During sampling, you start at high noise, take Langevin steps conditioned on that sigma, then decrease sigma and repeat so the score remains valid for the current noise regime as you approach the data distribution.

### Schedule density and step counts
More sigmas and steps increase compute and often give diminishing returns. Adjacent noise levels are similar, and the model can generalize across nearby sigmas when conditioned, so moderate schedules are a practical balance between quality and runtime.

### ODE-based sampling
An ODE solver replaces the Langevin loop. You integrate a deterministic probability-flow ODE from high noise to low noise using the learned score, instead of iterating stochastic Langevin steps at discrete sigmas.

### Deterministic vs stochastic sampling
Not necessarily. You can sample purely with the ODE (deterministic), purely with Langevin (stochastic), or combine them in predictor-corrector sampling where the ODE is the predictor and a few Langevin steps act as a corrector.
