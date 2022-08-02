import torch.nn as nn
import numpy as np
import torch.nn.functional as F
'''
modified to fit dataset size
ref: https://www.kaggle.com/code/drvaibhavkumar/alexnet-in-pytorch-cifar10-clas-83-test-accuracy/notebook
'''
NUM_CLASSES = 10
""" BN : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9066113"""
""" ref: https://github.com/soapisnotfat/pytorch-cifar10/blob/master/models/AlexNet.py"""
class AlexNet(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.layer0 = nn.Sequential(
            # Block 1
            # nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            # Block 2
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer3 = nn.Sequential(
            # Block 3
            # nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            # Block 4
            # nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            # Block 5
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        # self.features = nn.Sequential(
        #     # Block 1
        #     nn.BatchNorm2d(3),
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, ceil_mode=False),
        #     # Block 2
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, ceil_mode=False),
        #     # Block 3
        #     nn.BatchNorm2d(192),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # Block 4
        #     nn.BatchNorm2d(384),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # Block 5
        #     nn.BatchNorm2d(256),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, ceil_mode=False),
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear(384 * 4 * 4, 4096, bias=True),
            # nn.Linear(256 * 4 * 4, 4096, bias=True),
            nn.Linear(256 * 2 * 2, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes, bias=True)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.layer0(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # input(x.shape)
        # x = self.avgpool(x)
        # input(x.shape)
        # x = x.view(x.size(0), 384 * 4 * 4)
        # x = x.view(x.size(0), 256 * 4 * 4)
        x = x.view(x.size(0), 256 * 2 * 2)
        # input(x.shape)

        logits = self.classifier(x)
        # probas = F.softmax(logits, dim=1)
        return logits