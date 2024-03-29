import torch.nn as nn
import torch.nn.functional as F
import torch

class SquareNet(nn.Module):
    def __init__(self):
        super(SquareNet, self).__init__()
        self.linear1 = nn.Linear(1024, 1024)
        self.relu1 = nn.ReLU()
        self.sig1 = nn.Sigmoid()
        self.linear2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.sig2 = nn.Sigmoid()
        self.linear3 = nn.Linear(1024, 1024)
        self.relu3 = nn.ReLU()
        self.sig3 = nn.Sigmoid()
        self.linear4 = nn.Linear(1024, 1024)
        self.relu4 = nn.ReLU()
        self.sig4 = nn.Sigmoid()
        self.linear5 = nn.Linear(1024, 84)
        self.relu5 = nn.ReLU()
        self.sig5 = nn.Sigmoid()
        # self.linear1 = nn.Linear(120, 84)
        self.softmax = nn.Linear(84, 10)
        self.tanh = nn.Tanh()
        # self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.tanh(x)
        # x = self.avgpool(x)
        # x = self.conv2(x)
        # x = self.tanh(x)
        # x = self.avgpool(x)
        # x = self.conv3(x)
        # x = self.tanh(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.sig1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        # x = self.sig2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        # x = self.sig3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        # x = self.sig4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        # x = self.sig5(x)
        # x = self.tanh(x)
        x = self.softmax(x)
        return x