import torch
from torch import nn


class YuKai(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


yk = YuKai()
x = torch.tensor(1.0)
output = yk(x)
print(output)
