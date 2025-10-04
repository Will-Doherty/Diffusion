import torch
import torch.nn as nn
from torch.nn.functional import relu, mse_loss
from torchvision import transforms, datasets
from dataclasses import dataclass
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import subprocess
from tqdm import tqdm
from einops import einsum


# --- DATASET ---

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


# --- MODEL DEFINITION ---

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128)):
        super().__init__()
        f1, f2, fb = features
        self.down1 = DoubleConv(in_channels, f1)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(f1, f2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f2, fb)
        self.up2 = nn.ConvTranspose2d(fb, f2, 2, stride=2)
        self.conv2 = DoubleConv(f2 + f2, f2)
        self.up1 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
        self.conv1 = DoubleConv(f1 + f1, f1)
        self.out = nn.Conv2d(f1, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        b = self.bottleneck(self.pool2(d2))
        u2 = self.up2(b)
        x = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(x)
        x = self.conv1(torch.cat([u1, d1], dim=1))
        return self.out(x)


# --- TRAINING CODE ---

def sliced_score_matching(score_est):
    assert score_est.dim() == 4, "Tensor must be of shape (B, C, X, Y)"

    batch_size = score_est.shape[0]
    # n_channels = x.shape[1]
    n_channels = 3
    dim1 = score_est.shape[2]
    dim2 = score_est.shape[3]

    v = torch.randn_like(score_est)
    # normalize across all pixel values but have a separate normalization factor for each channel
    normalization_factor = torch.linalg.vector_norm(v, dim=(2, 3)).repeat(batch_size, n_channels, dim1, dim2)
    v = v / normalization_factor

    # now we need the hessian
    hessian = 
    
    term1 = v.T * hessian * v
    term2 = 0.5 * (v.T * score_est) ** 2
    
    loss = (term1 + term2).mean()
    return loss

@dataclass
class Config:
    lr = 1e-4


def train_score_matching():
    cfg = Config()
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    losses = []
    # for i, image_class_pair in tqdm(enumerate(train_loader)):
    for i, image_class_pair in enumerate(train_loader):
        x = image_class_pair[0]
        model.zero_grad(set_to_none=True)
        output = model(x)
        loss = sliced_score_matching(output)
        losses.append(loss)
        loss.backward()
        optimizer.step()

    # plt.plot(losses)
    # os.makedirs("outputs", exist_ok=True)
    # linux_path = os.path.abspath("outputs/losses.png")
    # plt.savefig(linux_path)
    # win_path = subprocess.check_output(["wslpath", "-w", linux_path]).decode().strip()
    # subprocess.run(
    #     ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", f"Start-Process -FilePath '{win_path}'"],
    #     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    # )
    # plt.close()


# --- INFERENCE CODE ---

def annealed_langevin_step(current_x: torch.Tensor, step_size: float) -> torch.Tensor:
    # TODO - parameterize this function and the one below with time
    noise = torch.randn((), dtype=torch.float64)
    return current_x + 0.5 * step_size * model(current_x) + torch.sqrt(torch.tensor(step_size, dtype=torch.float64)) * noise


def run_annealed_langevin_sampling(n_steps: int, initial_x: float, step_size: float) -> torch.Tensor:
    samples = torch.zeros(n_steps, dtype=torch.float64)
    current_x = torch.tensor(initial_x, dtype=torch.float64)
    samples[0] = current_x
    for i in range(1, n_steps):
        current_x = langevin_step(current_x, step_size)
        samples[i] = current_x
    return samples


if __name__ == "__main__":
    train_score_matching()