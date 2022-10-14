import torch.nn as nn
import torch


input = torch.randn(20, 100, 35, 45)
# m = nn.BatchNorm2d(100)
m = nn.BatchNorm2d(100, affine=False)

output = m(input)
print(output)

