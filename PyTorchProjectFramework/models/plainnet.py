import torch.nn as nn
import torch.nn.functional as F
import torch

class FcBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(FcBlock, self).__init__()
        self.fc1 = nn.Linear(
            in_planes, planes,bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.fc2 = nn.Linear(planes, planes, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Linear(in_planes, self.expansion*planes, bias=False),
                # nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.fc1(x)))
        out = F.relu(self.fc1(x))
        # out = self.bn2(self.fc2(out))
        out = self.fc2(out)
        # out += self.shortcut(x)
        out = F.relu(out)
        return out
class plainnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(plainnet, self).__init__()
        self.in_planes = 64
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32, 64, bias=False)
        # self.fc1 = nn.Linear(3*32*32, 64, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.flatten(x)
        # print(out.shape)
        # input()
        out = self.fc1(out)
        # print(out.shape)
        # out = self.conv1(x)
        # print(out.shape)
        # input()
        # out = self.bn1(out)
        out = F.relu(out)
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        # out = F.avg_pool2d(out, 4)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.linear(out)
        return out

def PlainNet18(num_classes=10):
    return plainnet(FcBlock, [2, 2, 2, 2],num_classes=num_classes)