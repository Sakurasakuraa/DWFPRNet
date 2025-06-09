import torch
from torch import nn
import torch.nn.functional as F

def convblock(in_, out_, ks, st=1, pad=0, dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias = False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

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


class CIE(nn.Module):
    def __init__(self, num_channels, ratio=8):
        super(CIE, self).__init__()
        self.conv_cross = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_cross = nn.BatchNorm2d(num_channels)

        self.eps = 1e-5

        self.conv_mask = nn.Conv2d(num_channels, 1, kernel_size=1)  # context Modeling

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // ratio, kernel_size=1),
            nn.LayerNorm([num_channels // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // ratio, num_channels, kernel_size=1)
        )
        self.initialize()

    def forward(self, in1):

        x = in1
        x = F.relu(self.bn_cross(self.conv_cross(x)))  # [B, C, H, W]

        context = (x.pow(2).sum((2, 3), keepdim=True) + self.eps).pow(0.5)  # [B, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)

        out = x * channel_add_term
        return out

    def initialize(self):
        weight_init(self)

class fuse(nn.Module):
    def __init__(self, in_channel):
        super(fuse, self).__init__()
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.bconv1 = convblock(2 * in_channel, in_channel, ks=3, pad=1)
        self.bconv2 = convblock(3 * in_channel, in_channel, ks=3, pad=1)
        self.cie = CIE(in_channel)

    def forward(self, in1, in2):

        in1_sa = self.sa1(in1) * in2
        in2_sa = self.sa2(in2) * in1

        in3 = torch.cat([in1_sa, in2_sa], dim=1)
        in3 = self.bconv1(in3)
        in3_sa_ca = self.cie(in3)
        in1_sa_ca = in1 * in3_sa_ca + in1
        in2_sa_ca = in2 * in3_sa_ca + in2
        in3_saca = in3 * in3_sa_ca


        f = torch.cat((in1_sa_ca, in2_sa_ca, in3_saca), dim=1)
        f = self.bconv2(f)

        return f