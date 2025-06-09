import torch
from torch import nn
import torch.nn.functional as F

def convblock(in_, out_, ks, st=1, pad=0,dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias = False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class cross_enhance(nn.Module):
    def __init__(self, in_channel):
        super(cross_enhance, self).__init__()
        self.rgb_CA = ChannelAttention(in_channel)
        self.d_CA = ChannelAttention(in_channel)
        self.t_CA = ChannelAttention(in_channel)
        self.rgb_SA = SpatialAttention()
        self.d_SA = SpatialAttention()
        self.t_SA = SpatialAttention()
        self.conv_rgb = convblock(2 * in_channel, in_channel, 3, st=1, pad=1)
        self.conv_d = convblock(2 * in_channel, in_channel, 3, st=1, pad=1)
        self.conv_t = convblock(2 * in_channel, in_channel, 3, st=1, pad=1)

    def forward(self, rgb, d, t):
        rgb_ca = rgb.mul(self.rgb_CA(rgb))
        d_ca = d.mul(self.d_CA(d))
        t_ca = t.mul(self.t_CA(t))

        rgb_ca_sa = self.rgb_SA(rgb_ca)
        d_ca_sa = self.d_SA(d_ca)
        t_ca_sa = self.t_SA(t_ca)

        #d、t增强rgb
        rgb_d = rgb_ca + rgb_ca.mul(d_ca_sa)
        rgb_t = rgb_ca + rgb_ca.mul(t_ca_sa)
        rgb_dt = torch.cat([rgb_d, rgb_t], dim=1)
        rgb_e = self.conv_rgb(rgb_dt)
        # rgb、t增强d
        d_rgb = d_ca + d_ca.mul(rgb_ca_sa)
        d_t = d_ca + d_ca.mul(t_ca_sa)
        d_rgbt = torch.cat([d_rgb, d_t], dim=1)
        d_e = self.conv_d(d_rgbt)
        # d、rgb增强t
        t_d = t_ca + t_ca.mul(d_ca_sa)
        t_rgb = t_ca + t_ca.mul(rgb_ca_sa)
        t_rgbd = torch.cat([t_rgb, t_d], dim=1)
        t_e = self.conv_rgb(t_rgbd)

        return rgb_e, d_e, t_e

