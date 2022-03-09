import torch

def tuple2torch(x):
    x1, x2 = x
    y = torch.cat((x1.unsqueeze(-1), x2.unsqueeze(-1)), -1)
    return y