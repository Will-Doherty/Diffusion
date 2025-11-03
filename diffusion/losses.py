import torch

def calculate_annealed_sm_objective_mnist(model, x, sigma):
    assert x.dim() in [3, 4], "Input x must be three or four-dimensional"
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.requires_grad_(True)
    x_perturbed = x + torch.randn_like(x) * sigma
    v = torch.randn_like(x_perturbed)
    energy = model(x_perturbed, sigma).sum()
    score = torch.autograd.grad(-energy, x, create_graph=True)[0]
    score_v_product = score * v
    loss1 = torch.sum(0.5 * score_v_product ** 2)
    Hv = torch.autograd.grad(score_v_product.sum(), x, create_graph=True)[0]
    loss2 = v * Hv
    return (loss1 + loss2).mean()
