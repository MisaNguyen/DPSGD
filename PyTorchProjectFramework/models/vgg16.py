import torch
import torch.nn as nn
# from torchsummary import summary
# ref: https://discuss.pytorch.org/t/vgg16-using-cifar10-not-converging/114693
# Defining model

class VGGNet(nn.Module):
    def __init__(self, in_channels, num_classes, arch):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(arch)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)))
        self.fcs = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)
        x = self.avgpool(x)
        # x = x.reshape(x.shape[0], -1)
        x = torch.flatten(x, 1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels

        for x in arch:

            if type(x) == int:

                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]

                in_channels = x

            elif x =='M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)
if __name__ == '__main__':
    arch = [64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M']
    model = VGGNet(in_channels=3, num_classes=10, arch=arch)
    print(model)