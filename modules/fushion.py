import torch
from torch import nn
import torch.nn.functional as F

def convblock(in_, out_, ks, st=1, pad=0, dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias = False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class fushion(nn.Module):
    def __init__(self, in_channel):
        super(fushion, self).__init__()
        self.rgbd_conv1 = convblock(in_channel, in_channel, ks=3, st=1, pad=1)
        self.rgbd_conv2 = convblock(3*in_channel, in_channel, ks=3, st=1, pad=1)
        self.rgbd_aap = nn.AdaptiveAvgPool2d(1)
        self.rgbd_amp = nn.AdaptiveMaxPool2d(1)

        self.rgbt_conv1 = convblock(in_channel, in_channel, ks=3, st=1, pad=1)
        self.rgbt_conv2 = convblock(3*in_channel, in_channel, ks=3, st=1, pad=1)
        self.rgbt_aap = nn.AdaptiveAvgPool2d(1)
        self.rgbt_amp = nn.AdaptiveMaxPool2d(1)

        self.dt_conv1 = convblock(in_channel, in_channel, ks=3, st=1, pad=1)
        self.dt_conv2 = convblock(3*in_channel, in_channel, ks=3, st=1, pad=1)
        self.dt_aap = nn.AdaptiveAvgPool2d(1)
        self.dt_amp = nn.AdaptiveMaxPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, d, t):
        rgbd = rgb.mul(d)
        rgbt = rgb.mul(t)
        dt = d.mul(t)

        vd1 = self.rgbd_conv1(rgbd)
        vt2 = self.rgbt_conv1(rgbt)
        dt3 = self.dt_conv1(dt)

        rgbd_aap = self.rgbd_aap(rgbd)
        rgbd_amp = self.rgbd_amp(rgbd)
        rgbd_w = self.sigmoid(rgbd_aap+rgbd_amp)
        dt1 = dt3 * rgbd_w + dt3
        vt1 = vt2 * (1-rgbd_w) + vt2
        f1 = self.rgbd_conv2(torch.cat([vd1, dt1, vt1], dim=1))

        rgbt_aap = self.rgbt_aap(rgbt)
        rgbt_amp = self.rgbt_amp(rgbt)
        vt_w = self.sigmoid(rgbt_aap+rgbt_amp)
        vd2 = vd1 * vt_w + vd1
        dt2 = dt3 * (1-vt_w) + dt3
        f2 = self.rgbt_conv2(torch.cat([vd2, dt2, vt2], dim=1))

        dt_aap = self.dt_aap(dt)
        dt_amp = self.dt_amp(dt)
        dt_w = self.sigmoid(dt_aap+dt_amp)
        vd3 = vd1 * dt_w + vd1
        vt3 = vt2 * (1-dt_w) + vt2
        f3 = self.dt_conv2(torch.cat([vd3, dt3, vt3], dim=1))

        fusion_feature = f1 + f2 + f3

        return fusion_feature
