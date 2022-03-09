import sys
sys.path.append('../Classes')
from util import *
import math
import torch

# Variables

m = 85000
c = 6
aL = 30
aD = 2
Tmin = 40000
Tmax = 80000
omin = -22.5 * (math.pi / 180)
omax = 22.5 * (math.pi / 180)
Vmin = 180
Vmax = 240
ymin = -22.5 * (math.pi / 180)
ymax = 22.5 * (math.pi / 180)
g = 9.8

# Functions

# System dynamics
def f(x, u):
    V_, y_ = x.T
    u1_, u2_ = u.T
    Vt = - aD * V_ ** 2 / m - g * torch.sin(y_) + 1 / m * u1_
    yt = aL * y_ * (1 - c * V_) / m - g * torch.cos(y_) / V_ + aL * c * V_ / m * u2_
    xt_  = tuple2torch((Vt,yt))
    return xt_

# Optimal controls
def u(x):
    V_, y_ = x.T

    u1 = torch.where(V_ < (Vmin + Vmax) / 2, Tmax, Tmin)
    u2 = torch.where(y_ < (ymin + ymax) / 2, omax, omin)

    u_ = tuple2torch((u1,u2))
    
    return u_

# Loss function
adj = (ymax - ymin) / (Vmax - Vmin) # scaling factor
def l(x):
    x = x.T.unsqueeze(-1)
    diff = torch.cat((adj*(x[0] - Vmin), x[1] - ymin, adj*(Vmax - x[0]), ymax - x[1]), -1)
    loss = torch.min(diff, -1).values
    return loss
    