import torch
v1 = torch.arange(1., 5.).long()
v2 = torch.arange(1., 4.).long()
torch.ger(v1, v2)
