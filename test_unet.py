import torch
from score_matching_mnist import UNet

def test_unet_output_shape():
    model = UNet()
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 1, 28, 28)