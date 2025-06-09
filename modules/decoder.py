import torch
from torch import nn
from modules.aspp815 import aspp
import torch.nn.functional as F

def convblock(in_, out_, ks, st=1, pad=0, dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias = False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

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

class upmodule_sa(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(upmodule_sa, self).__init__()
        self.conv1 = convblock(in_channel, out_channel, ks=3, st=1, pad=1)
        self.conv2 = convblock(2 * out_channel, out_channel, ks=3, st=1, pad=1)

        # self.up2 = F.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        self.sa = SpatialAttention()

    def forward(self, d5, d4):
        size = d4.size()[2:]
        d5_up2_conv = self.conv1(F.interpolate(d5, size, mode='bilinear', align_corners=True))
        # d5_up2 = self.up2(d5)
        d5_up2_conv_sa = self.sa(d5_up2_conv)

        d4_1 = d4 * d5_up2_conv_sa + d4
        a = torch.cat([d4_1, d5_up2_conv], dim=1)
        d = self.conv2(a)

        return d

class upmodule_ca(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(upmodule_ca, self).__init__()
        self.conv1 = convblock(in_channel, out_channel, ks=3, st=1, pad=1)
        self.conv2 = convblock(2 * out_channel, out_channel, ks=3, st=1, pad=1)

        # self.up2 = F.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        self.ca = ChannelAttention(out_channel)

    def forward(self, d5, d4):
        size = d4.size()[2:]
        d5_up2_conv = self.conv1(F.interpolate(d5, size, mode='bilinear', align_corners=True))
        # d5_up2 = self.up2(d5)
        d5_up2_conv_ca = self.ca(d5_up2_conv)

        d4_1 = d4 * d5_up2_conv_ca + d4
        a = torch.cat([d4_1, d5_up2_conv], dim=1)
        d = self.conv2(a)

        return d

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.dblock4 = upmodule_ca(512, 512)
        self.dblock3 = upmodule_ca(512, 256)
        self.dblock2 = upmodule_sa(256, 128)
        self.dblock1 = upmodule_sa(128, 64)

        self.aspp1 = aspp(512)
        self.aspp2 = aspp(512)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.s3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            self.upsample8,
            nn.Conv2d(256, 1, 3, stride=1, padding=1)
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            self.upsample4,
            nn.Conv2d(128, 1, 3, stride=1, padding=1)
        )
        self.s1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            self.upsample2,
            nn.Conv2d(64, 1, 3, stride=1, padding=1)
        )
        self.s = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, 1, 3, stride=1, padding=1)
        )



    def forward(self, feature_list):
        # xsize = feature_list[0].size()[2:]
        d5 = self.aspp1(feature_list[4])
        d4 = self.dblock4(d5, feature_list[3])
        d4 = self.aspp2(d4)
        d3 = self.dblock3(d4, feature_list[2])
        d2 = self.dblock2(d3, feature_list[1])
        d1 = self.dblock1(d2, feature_list[0])

        s3 = self.s3(d4)
        s2 = self.s2(d3)
        s1 = self.s1(d2)
        s = self.s(d1)

        return s, s1, s2, s3



