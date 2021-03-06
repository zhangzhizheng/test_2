import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, num=0):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.num = num

    def forward(self, x):
        time_start = time.time()

        # if(self.num == 0):
        #     with open('/home/test_2/time/convo_0_'+str(self.num)+'.pkl', 'ab') as f:
        #         #print('a')
        #         pickle.dump(x, f)
        conv_1 = self.conv1(x)
        time_stop = time.time()
        print('convo_1_'+str(self.num)+':', time_stop-time_start)
        with open('/home/test_2/time/m_convo_1_'+str(self.num)+'.pkl', 'wb') as f:
            #print('a')
            pickle.dump(conv_1, f)
        out = F.relu(self.bn1(conv_1))
        time_start = time.time()
        conv_2 = self.conv2(out)
        time_stop = time.time()
        print('convo_2_'+str(self.num)+':', time_stop-time_start)
        with open('/home/test_2/time/m_convo_2_'+str(self.num)+'.pkl', 'wb') as f:
            #print('a')
            pickle.dump(conv_1, f)
        out = F.relu(self.bn2(conv_2))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        num = 0
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride, num))
            in_planes = out_planes
            num += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        with open('/home/test_2/time/m_convo_0_0_0.pkl', 'wb') as f:
            #print('a')
            pickle.dump(x, f)
        time_start = time.time()
        conv_1 = self.conv1(x)
        time_stop = time.time()
        with open('/home/test_2/time/m_convo_0_1_0.pkl', 'wb') as f:
            #print('a')
            pickle.dump(conv_1, f)
        print('convo_0_0:', time_stop-time_start)
        out = F.relu(self.bn1(conv_1))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out