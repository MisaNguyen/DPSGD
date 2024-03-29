import torch.nn as nn
import torch.nn.functional as F
import torch

class nor_LeNet(nn.Module):
    def __init__(self):
        super(nor_LeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 6,
                      kernel_size = 5, stride = 1, padding = 0),
            nn.BatchNorm2d(6),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 6, out_channels = 16,
                      kernel_size = 5, stride = 1, padding = 0),
            nn.BatchNorm2d(16),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 120,
                      kernel_size = 5, stride = 1, padding = 0),
            nn.BatchNorm2d(120),
        )
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.tanh(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x