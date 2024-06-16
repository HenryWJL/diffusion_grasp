from einops.layers.torch import Rearrange
import torch
from torch import nn


x = torch.rand(2, 3, 4)
m1 = nn.Linear(4, 8)
y = m1(x)
print(y[..., :4])
m2 = Rearrange('b n (d c) -> c b n d', d=4, c=2)
print(m2(y).chunk(2, dim=0)[0])