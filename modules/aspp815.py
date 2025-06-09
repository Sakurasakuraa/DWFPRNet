import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

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


class aspp(nn.Module):
    def __init__(self, in_channel):
        super(aspp, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, in_channel, 1, 1)
        # k=1 s=1 no pad


        self.r_block1 = nn.Conv2d(in_channel, in_channel, 3, 1,padding=1, dilation=1)
        self.r_block3 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=3, dilation=3)
        self.r_block5 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=5, dilation=5)

        self.ca1 = ChannelAttention(in_channel)
        self.ca2 = ChannelAttention(in_channel)

        self.r_conv_output = nn.Conv2d(in_channel * 3, in_channel, 3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.Bconv = BasicConv2d(in_channel, in_channel, 3, padding=1)

    def forward(self, in1):

        image_features = self.mean(in1)
        image_features = self.conv(image_features)
        weight = self.sigmoid(image_features)


        r_block1 = self.r_block1(in1)
        r1_ca = self.ca1(r_block1)
        r_block3 = self.r_block3(in1 * r1_ca)
        r3_ca = self.ca2(r_block3)
        r_block5 = self.r_block5(in1 * r3_ca)

        r_net = self.r_conv_output(torch.cat([r_block1, r_block3, r_block5], dim=1))
        out = r_net * weight

        out_final = self.Bconv(out)

        return out_final
