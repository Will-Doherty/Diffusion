import torch.nn as nn
from torch.nn.functional import gelu

class ScoreFnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1, bias=False)

        nn.init.normal_(self.layer1.weight, std=0.02)
        nn.init.normal_(self.layer2.weight, std=0.02)
        nn.init.normal_(self.layer3.weight, std=0.02)

    def forward(self, x):
        x = gelu(self.layer1(x))  # swapped to gelu because relu has 0s everywhere in its Hessian
        x = gelu(self.layer2(x))
        x = self.layer3(x)
        return x

