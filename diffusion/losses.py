import torch

def calculate_sliced_sm_objective_mnist(model, x):
    assert x.dim() in [3, 4], "Input x must be three or four-dimensional"
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.requires_grad_(True)
    v = torch.randn_like(x)
    energy = model(x).sum()
    score = torch.autograd.grad(-energy, x, create_graph=True)[0]
    score_v_product = score * v
    loss1 = torch.sum(0.5 * score_v_product ** 2)
    Hv = torch.autograd.grad(score_v_product.sum(), x, create_graph=True)[0]
    loss2 = v * Hv
    return (loss1 + loss2).mean()
