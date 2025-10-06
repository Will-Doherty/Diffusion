import matplotlib.pyplot as plt
import torch

def plot_gm_sampling_result(inference_samples, gm):
    x_range = torch.linspace(-5, 8, 100, dtype=torch.float64)
    y_range = torch.linspace(-5, 8, 100, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)

    density = gm.get_density(grid)

    plt.figure(figsize=(8, 8))
    plt.contourf(grid_x.numpy(), grid_y.numpy(), density.numpy(), levels=20, cmap='viridis')
    plt.colorbar(label='Density')

    samples_np = inference_samples.detach().numpy()
    plt.scatter(samples_np[:, 0], samples_np[:, 1], c='r', label='Langevin Samples', marker='x', s=10, alpha=0.5)

    plt.title('True Distribution and Langevin Samples')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_mnist_sampling_result(inference_samples, gm):
    pass