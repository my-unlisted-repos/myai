import torch
from torch import nn
from myai import nn as mynn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, ndim = 2):
        super().__init__()
        self.conv1 = mynn.convnd(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, ndim=ndim)
        self.bn1 = nn.BatchNorm2d(out_channels) if ndim == 2 else nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = mynn.convnd(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, ndim = ndim)
        self.bn2 = nn.BatchNorm2d(out_channels) if ndim == 2 else nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                mynn.convnd(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, ndim = ndim),
                nn.BatchNorm2d(out_channels) if ndim == 2 else nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels, step = 32, num_classes=10, ndim = 2):
        super().__init__()
        self.in_channels = step
        self.conv1 = mynn.convnd(in_channels, step, kernel_size=3, stride=1, padding=1, bias=False, ndim = ndim)
        self.bn1 = nn.BatchNorm2d(step) if ndim == 2 else nn.BatchNorm1d(step)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = mynn.maxpoolnd(kernel_size=3, stride=2, padding=1, ndim = ndim)

        self.layer1 = self._make_layer(BasicBlock, step, 2, stride=1, ndim=ndim)
        self.layer2 = self._make_layer(BasicBlock, step*2, 2, stride=2, ndim=ndim)
        self.layer3 = self._make_layer(BasicBlock, step*4, 2, stride=2, ndim=ndim)
        self.layer4 = self._make_layer(BasicBlock, step*8, 2, stride=2, ndim=ndim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if ndim == 2 else nn.AdaptiveAvgPool1d((1,))
        self.fc = nn.Linear(step*8, num_classes)

        self.ndim = ndim

    def _make_layer(self, block, out_channels, num_blocks, stride, ndim):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_ in strides:
            layers.append(block(self.in_channels, out_channels, stride_, ndim))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.ndim == 1 and x.ndim == 2: x = x.unsqueeze(1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
