import torch
import torch.nn as nn


class SeqBottle(nn.Module):
    def __init__(self, in_channel, out_channel, stride=False, inter_bias=False, inter_BN=False):
        super(SeqBottle, self).__init__()
        if inter_BN: inter_bias = False
        self.conv1 = nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, padding=1, bias=inter_bias)
        weight = self._init_weight(in_channel, in_channel) # in_channel * 2, in_channel
        conv1_weight = nn.ZeroPad2d(1)(weight.view(in_channel * 2, in_channel, 1, 1))
        self.conv1.weight = nn.Parameter(conv1_weight)


        self.conv2 = nn.Conv2d(in_channel, in_channel * 4, kernel_size=1, bias=inter_bias)
        weight = self._init_weight(in_channel, in_channel * 2) # in_channel * 4, in_channel
        conv2_weight = weight.view(in_channel * 4, in_channel, 1, 1)
        self.conv2.weight = nn.Parameter(conv2_weight.contiguous())


        self.conv3 = nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, bias=False)
        weight = torch.randn(out_channel, in_channel * 4)
        weight[:in_channel] = conv2_weight[:,:,0,0].transpose(0, 1) / 2 ** .5
        for k in range(in_channel, out_channel):
            r = weight[k] - weight[:k].T @ (weight[:k] @ weight[k])
            weight[k] = r / r.norm()
        conv3_weight = weight.div(2**.5).view(out_channel, -1, 1, 1)
        self.conv3.weight = nn.Parameter(conv3_weight.contiguous())

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.inter_BN = inter_BN
        self.stride = stride

        if inter_bias:
            nn.init.zeros_(self.conv1.bias)
            nn.init.zeros_(self.conv2.bias)
        if inter_BN:
            self.inter_bn1 = nn.BatchNorm2d(in_channel * 2)
            self.inter_bn2 = nn.BatchNorm2d(in_channel * 4)

        if stride:
            self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        # 3x3
        x = self.conv1(x)
        if self.inter_BN:
            x = self.inter_bn1(x)
        if self.stride:
            x = self.pool(x)
        x = x.relu_()
        weight = self.conv1.weight[:,:,1:2,1:2].transpose(0,1)
        x = nn.functional.conv2d(x, weight=weight)
        x = self.bn1(x)

        # 1x1
        x = self.conv2(x)
        if self.inter_BN:
            x = self.inter_bn2(x)
        x = x.relu_()
        x = self.conv3(x)
        x = self.bn2(x)
        return x

    def _init_weight(self, in_channel, mid_channel):
        assert mid_channel >= in_channel
        W = torch.randn(mid_channel, in_channel)
        nn.init.orthogonal_(W)
        weight = torch.cat([W, -W], dim=0)
        return weight



class plainnet(nn.Module):
    def __init__(self, num_layers, num_classes, inter_bias, inter_BN):
        super(plainnet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.MaxPool2d(3, 2, 1))

        self.inter_bias = inter_bias
        self.inter_BN = inter_BN

        self.layer1 = self.__make_layer__(64, 128, num_layers[0])
        self.layer2 = self.__make_layer__(128, 256, num_layers[1], stride=True)
        self.layer3 = self.__make_layer__(256, 512, num_layers[2], stride=True)
        self.layer4 = self.__make_layer__(512, 2048, num_layers[3], stride=True)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes))

    def __make_layer__(self, in_channels, out_channels, num_blocks, stride=False):
        layers = [SeqBottle(in_channels, in_channels,
                            stride, self.inter_bias, self.inter_BN)]
        for k in range(1, num_blocks):
            out_channel = in_channels if k != num_blocks -1 else out_channels
            layers += [SeqBottle(in_channels, out_channel,
                                 False, self.inter_bias, self.inter_BN)]
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x



def SeqNet50(num_classes=1000, inter_bias=False, inter_BN=False):
    return plainnet([3, 4, 6, 3], num_classes, inter_bias, inter_BN)

def SeqNet101(num_classes=1000, inter_bias=False, inter_BN=False):
    return plainnet([3, 4, 23, 3], num_classes, inter_bias, inter_BN)