# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/11 0011, matt '


import torch
import torch.nn as nn
from model.utils import *


class DNCNN(nn.Module):
    def __init__(self, verbose=False):
        super(DNCNN, self).__init__()
        self.verbose = verbose

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classify = nn.Sequential(
            nn.Linear(9216, 2048),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(2048, 2)
        )
        weights_init_kaiming(self)

    def forward(self, x):
        x = self.layer1(x)
        if self.verbose:
            print("layer1 size: ", x.size())
        x = self.layer2(x)
        if self.verbose:
            print("layer1 size: ", x.size())
        x = self.layer3(x)
        if self.verbose:
            print("layer3 size: ", x.size())
        x = self.pool1(x)
        if self.verbose:
            print("pool1 size: ", x.size())
        x = self.layer4(x)
        if self.verbose:
            print("layer4 size: ", x.size())
        x = self.layer5(x)
        if self.verbose:
            print("layer5 size: ", x.size())
        x = self.pool2(x)
        if self.verbose:
            print("pool2 size: ", x.size())
        x = self.layer6(x)
        if self.verbose:
            print("layer6 size: ", x.size())
        x = self.layer7(x)
        if self.verbose:
            print("layer7 size: ", x.size())
        x = self.layer8(x)
        if self.verbose:
            print("layer8 size: ", x.size())

        x = x.view(x.size(0), -1)
        if self.verbose:
            print("view size: ", x.size())
        x = self.classify(x)
        if self.verbose:
            print("classify size: ", x.size())
        return x


if __name__ == "__main__":
    net = DNCNN(verbose=True)
    x = torch.randn((1, 3, 48, 48))
    y = net(x)

