# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/11 0011, matt '


import torch
import torch.nn as nn
from model.utils import *


class Mul_ConvNet(nn.Module):
    def __init__(self, verbose=False):
        super(Mul_ConvNet, self).__init__()

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(6, 96, kernel_size=11, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.full6 = nn.Linear(9216, 4096)
        self.full7 = nn.Linear(4096, 2048)
        self.full8 = nn.Linear(2048, 2)
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.verbose = verbose

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.verbose:
            print("conv1 size: ", x.size())
        x = self.pool1(x)
        if self.verbose:
            print("pool1 size: ", x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.verbose:
            print("conv2 size: ", x.size())
        x = self.pool2(x)
        if self.verbose:
            print("pool2 size: ", x.size())
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        if self.verbose:
            print("conv3 size: ", x.size())
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        if self.verbose:
            print("conv4 size: ", x.size())
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        if self.verbose:
            print("conv5 size: ", x.size())
        x = self.pool5(x)
        if self.verbose:
            print("pool5 size: ", x.size())
        out1 = x.view(x.size(0), -1)
        if self.verbose:
            print("out1 size: ", out1.size())
        out1 = self.full6(out1)
        if self.verbose:
            print("full6 size: ", out1.size())
        out1 = self.full7(out1)
        if self.verbose:
            print("full6 size: ", out1.size())
        out1 = self.full8(out1)
        if self.verbose:
            print("full6 size: ", out1.size())

        out2 = self.deconv1(x)
        out2 = self.bn6(out2)
        out2 = self.relu(out2)
        if self.verbose:
            print("deconv1 size: ", out2.size())
        out2 = self.deconv2(out2)
        out2 = self.bn7(out2)
        out2 = self.relu(out2)
        if self.verbose:
            print("deconv2 size: ", out2.size())
        out2 = self.deconv3(out2)
        out2 = self.bn8(out2)
        out2 = self.relu(out2)
        if self.verbose:
            print("deconv3 size: ", out2.size())
        out2 = self.deconv4(out2)
        if self.verbose:
            print("deconv4 size: ", out2.size())
        return out1, out2


if __name__ == "__main__":

    print(nn.Sequential(*list(net.children()))[0])
    net = Mul_ConvNet(verbose=True)
    x = torch.randn((1, 6, 227, 227))
    y1, y2 = net(x)
