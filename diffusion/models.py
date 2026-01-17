import torch
import torch.nn as nn

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
        self.features = features
        self.sigma_embed1 = nn.Sequential(nn.Linear(1, f1), nn.SiLU(), nn.Linear(f1, f1))
        self.sigma_embed2 = nn.Sequential(nn.Linear(1, f2), nn.SiLU(), nn.Linear(f2, f2))
        self.sigma_embedb = nn.Sequential(nn.Linear(1, fb), nn.SiLU(), nn.Linear(fb, fb))
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

    def _sigma_to_tensor(self, sigma, x):
        if sigma is None:
            return None
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)
        if sigma.dim() == 0:
            sigma = sigma[None]
        if sigma.dim() == 1:
            sigma = sigma[:, None]
        if sigma.shape[0] == 1 and x.shape[0] > 1:
            sigma = sigma.expand(x.shape[0], 1)
        return sigma

    def forward(self, x, sigma=None):
        sigma = self._sigma_to_tensor(sigma, x)
        d1 = self.down1(x)
        if sigma is not None:
            d1 = d1 + self.sigma_embed1(sigma).view(x.shape[0], self.features[0], 1, 1)
        d2 = self.down2(self.pool1(d1))
        if sigma is not None:
            d2 = d2 + self.sigma_embed2(sigma).view(x.shape[0], self.features[1], 1, 1)
        b = self.bottleneck(self.pool2(d2))
        if sigma is not None:
            b = b + self.sigma_embedb(sigma).view(x.shape[0], self.features[2], 1, 1)
        u2 = self.up2(b)
        x = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(x)
        x = self.conv1(torch.cat([u1, d1], dim=1))
        return self.out(x)
