
import torch.nn as nn
""" Ref (opacus's modules fixers): https://opacus.ai/tutorials/guide_to_module_validator
def _batchnorm_to_groupnorm(module) -> nn.GroupNorm:

Converts a BatchNorm ``module`` to GroupNorm module.
This is a helper function.
Args:
module: BatchNorm module to be replaced
Returns:
GroupNorm module that can replace the BatchNorm module provided
Notes:
A default value of 32 is chosen for the number of groups based on the
paper *Group Normalization* https://arxiv.org/abs/1803.08494

return nn.GroupNorm(
    min(32, module.num_features), module.num_features, affine=module.affine
)
"""

class GN_BN_convnet(nn.Module):
    def __init__(self, num_classes):
        super(GN_BN_convnet, self).__init__()

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
            nn.GroupNorm(min(32, 32), 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(32, 64), 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(32, 64), 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(32, 128), 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.softmax_layer = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128, num_classes, bias=True),
        )
    def forward(self, x):
        # x = self.layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        # input()
        x = self.softmax_layer(x)
        return x