import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def convblock(in_, out_, ks, st=1, pad=0, dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias=False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class quality(nn.Module):
    def __init__(self, in_channel):
        super(quality, self).__init__()
        self.dgap = nn.AdaptiveAvgPool2d(1)
        self.tgap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)

        self.conv3 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)

        self.bconv = convblock(2 * in_channel, in_channel, ks=3, st=1, pad=1)

    def forward(self, rd, rt, d, t):
        _, c, _, _ = d.size()
        d_w = self.sigmoid(self.dgap(d))
        d_sum = torch.sum(d_w, axis=1)
        d_mean = d_sum / c

        rd_w1 = rd * d_w
        rt_w1 = rt * (1 - d_w)
        x1, _ = torch.max(rd_w1, dim=1, keepdim=True)
        x1 = self.conv2(self.conv1(x1))
        mask1 = torch.sigmoid(x1)

        f1 = rd_w1 + rt_w1 * mask1 + rt_w1

        t_w = self.sigmoid(self.tgap(t))
        t_sum = torch.sum(t_w, axis=1, keepdims=True)
        t_mean = t_sum / c

        rt_w2 = rt * t_w
        rd_w2 = rd * (1 - t_w)
        x2, _ = torch.max(rt_w2, dim=1, keepdim=True)
        x2 = self.conv4(self.conv3(x2))
        mask2 = torch.sigmoid(x2)

        f2 = rd_w2 + rd_w2 * mask2 + rt_w2

        feature = torch.cat((f1 + f2, f1 * f2), dim=1)
        feature = self.bconv(feature)

        return feature
