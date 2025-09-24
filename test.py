import torch
from torch.func import hessian, functional_call
from sliced_score_matching import ScoreFnNet

model = ScoreFnNet()
loss = 5.0
torch.func.hessian()