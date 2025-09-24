= Code structure

- _langevin_sampling_gm.py_: Demonstrates Langevin sampling from a 1D Gaussian mixture distribution with known score.
- _score_matching_gm.py_: The score function of a 1D Gaussian mixture is learned by a simple MLP. In this case, we have access to the ground-truth score during training. We then sample from the trained model using Langevin sampling.
- _sliced_score_matching_gm.py_: Demonstrates Langevin sampling from a 2D Gaussian mixture with score learned by a simple MLP. Implements the basic score matching objective function, as well as sliced score matching.


= Sliced Score Matching

The score matching objective is

$ J(theta) = 1/2 E_(p_"data") [
  ||nabla_x log p_"data" (x)
  - nabla_x log p_theta (x)||_2^2
] $

Since $nabla_x log p_"data" (x)$ is unknown, we construct an equivalent formulation

$ J(theta) = E_(p_"data") [
  "tr"(nabla_x^2 log p_theta (x))
  + 1/2 ||nabla_x log p_theta (x)||_2^2
] + "const" $

This avoids the dependence on $nabla_x log p_"data" (x)$, but still requires us to compute the Hessian $nabla_x^2$, which is impractical with high dimensionality. _Sliced score matching_ proposes to project the scores onto random directions, turning the vector fields of scores into scalar fields. Doing this yields the following objective

$ J(theta) = E_(p_"data") [
  v^T nabla_x^2 log p_theta (x) v
  + 1/2 (v^T nabla_x log p_theta (x))^2
] + "const" $

