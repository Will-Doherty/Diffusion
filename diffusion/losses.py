import torch


def calculate_sm_objective(model, x):
    # TODO: add batching
    x = x.squeeze(0).requires_grad_()
    print(x.shape)
    energy = model(x)
    print(energy.shape)
    score = -torch.autograd.grad(energy, x, create_graph=True)[0]
    loss1 = 0.5 * (score ** 2).sum(dim=-1)
    loss2 = 0.0
    D = x.shape[-1]
    for i in range(D):
        gi = torch.autograd.grad(score[..., i].sum(), x, create_graph=True, retain_graph=True)[0][..., i]
        loss2 = loss2 + gi
    return (loss1 + loss2).mean()

# def calculate_sliced_sm_objective(model, x):
#     x = x.squeeze(0).requires_grad_(True)
#     v = torch.randn_like(x)

#     y = model(x).sum()
#     score = torch.autograd.grad(y, x, create_graph=True)[0]
#     Hv = torch.autograd.grad((score * v).sum(), x, create_graph=True)[0]

#     first_summand = (v * Hv).sum()
#     second_summand = 0.5 * ((v * score).sum())**2
#     return first_summand + second_summand

def calculate_sm_objective_mnist(model, x):
    """
    Calculates the score matching objective for a batch of images (e.g., MNIST).
    This uses the Hutchinson's trace estimator for the trace of the Jacobian of the score,
    which is computationally efficient for high-dimensional data.
    """
    x.requires_grad_(True)
    
    energy = model(x)
    
    if energy.dim() > 1 and energy.shape[1] == 1:
        energy = energy.squeeze(1)

    score = -torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
    
    loss1 = 0.5 * (score.view(x.shape[0], -1)**2).sum(dim=1)
    
    v = torch.randn_like(x)
    grad_s_v = torch.autograd.grad((score * v).sum(), x, create_graph=True)[0]
    loss2 = (v * grad_s_v).view(x.shape[0], -1).sum(dim=1)
    
    total_loss = (loss1 + loss2)
    
    return total_loss.mean()