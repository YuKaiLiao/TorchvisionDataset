import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data", False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class YuKai(nn.Module):
    def __init__(self):
        super(YuKai, self).__init__()
        self.conv1 = Conv2d(3, 6, 3, 2, 0)

    def forward(self, x):
        x = self.conv1(x)
        return x


yk = YuKai()
# print(yk)

writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)

    output = yk(imgs)
    output = torch.reshape(output, [-1, 3, 30, 30])
    writer.add_images("output", output, step)

    step = step + 1
