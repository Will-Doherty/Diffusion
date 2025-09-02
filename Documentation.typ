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

