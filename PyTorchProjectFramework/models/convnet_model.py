
import torch.nn as nn


class convnet(nn.Module):
    def __init__(self, num_classes):
        super(convnet, self).__init__()

        # self.layers = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(start_dim=1, end_dim=-1),
        #     nn.Linear(128, num_classes, bias=True),
        # )
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.softmax_layer = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            # nn.Linear(128,1),
            nn.Linear(128, num_classes, bias=True),
            # nn.Linear(1, num_classes, bias=True),
        )
        # self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        # self.test_layer1 = nn.Linear(128,1)
        # self.test_layer2 = nn.Linear(1,1)
        # self.softmax_layer = nn.Linear(1, num_classes, bias=True)
    def forward(self, x):
        # x = self.layers(x)
        # print("x1=")
        # print(x.shape)
        x = self.layer1(x)
        # print("x1=")
        # print(x.shape)
        x = self.layer2(x)
        # print("x2=")
        # print(x.shape)
        x = self.layer3(x)
        # print("x3=")
        # print(x.shape)
        x = self.layer4(x)
        # print("x4=")
        # print(x.shape)
        # print(x.shape)
        # input()
        x = self.softmax_layer(x)
        # x= self.flatten(x)
        # x= self.test_layer1(x)
        # x= self.test_layer2(x)
        # x= self.softmax_layer(x)
        # print("x5=")
        # print(x.shape)
        # input()
        return x