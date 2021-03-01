'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride):
        super(Block, self).__init__()
        if(stride == 1 or stride == 3):
            if(in_planes == out_planes):
                self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=stride, stride=1, padding=0, bias=False)
                self.bn1 = nn.BatchNorm2d(out_planes)
            elif(in_planes != out_planes):
                self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
                self.bn1 = nn.BatchNorm2d(out_planes)              
        elif(stride == 2):
            # print(1)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        # self.dropout_1 = nn.Dropout(0.2)
        # self.dropout_2 = nn.Dropout(0.5)

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        # out = self.dropout_1(out)
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (64,2),128, (128,2), 256, (256,2),   512] # , 512, 512, 512, 512, 512, 1024, 1024]

    def __init__(self, num_classes=60):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # m = nn.AdaptiveMaxPool2d(32)
        # x = m(x)
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        # print(out.shape)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
