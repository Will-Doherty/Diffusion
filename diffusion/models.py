import torch
import torch.nn as nn


# -- U-NET --

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


# -- REFINE-NET --

class RCU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        r = x
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + r

class CRP(nn.Module):
    def __init__(self, channels, stages=2):
        super().__init__()
        self.stages = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(stages)])
        self.pool = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        out = x
        for conv in self.stages:
            out = self.pool(out)
            out = conv(out)
            x = x + out
        return x

class RefineBlock(nn.Module):
    def __init__(self, in_high, in_low, out_c):
        super().__init__()
        self.has_low = in_low is not None
        self.rcu_high = RCU(in_high)
        self.adapt_high = nn.Conv2d(in_high, out_c, 1)
        if self.has_low:
            self.rcu_low = RCU(in_low)
            self.adapt_low = nn.Conv2d(in_low, out_c, 1)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.crp = CRP(out_c)
        self.rcu_out = RCU(out_c)

    def forward(self, high, low=None):
        h = self.adapt_high(self.rcu_high(high))
        if self.has_low and low is not None:
            l = self.adapt_low(self.rcu_low(low))
            h = self.up(h)
            x = h + l
        else:
            x = h
        x = self.crp(x)
        x = self.rcu_out(x)
        return x

class RefineNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128)):
        super().__init__()
        f1, f2, fb = features
        self.enc1 = DoubleConv(in_channels, f1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(f1, f2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(f2, fb)
        self.refine2 = RefineBlock(fb, f2, f2)
        self.refine1 = RefineBlock(f2, f1, f1)
        self.out = nn.Conv2d(f1, out_channels, 1)

    def forward(self, x):
        l1 = self.enc1(x)
        l2 = self.enc2(self.pool1(l1))
        l3 = self.enc3(self.pool2(l2))
        r2 = self.refine2(l3, l2)
        r1 = self.refine1(r2, l1)
        return self.out(r1)