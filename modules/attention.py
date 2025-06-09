import torch
from torch import nn

def convblock(in_, out_, ks, st=1, pad=0, dila=1):
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

class attention(nn.Module):
    def __init__(self, in_channel):
        super(attention, self).__init__()

        self.rgb_CA = ChannelAttention(in_channel)
        self.d_CA = ChannelAttention(in_channel)
        self.t_CA = ChannelAttention(in_channel)

        self.query_rgb = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.key_rgb = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.value_rgb = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.gamma_1 = nn.Parameter(torch.zeros(1))

        self.query_d = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.key_d = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.value_d = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.gamma_2 = nn.Parameter(torch.zeros(1))

        self.query_t = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.key_t = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.value_t = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.gamma_3 = nn.Parameter(torch.zeros(1))

        self.conv1 = convblock(3 * in_channel, in_channel, ks=3, st=1, pad=1)
        self.conv2 = convblock(2 * in_channel, in_channel, ks=3, st=1, pad=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb, d, t, f5):
        rgb = self.rgb_CA(rgb)*rgb
        d = self.d_CA(d)*d
        t = self.t_CA(t)*t

        batchsize, C, height, width = rgb.size()

        rgb_q = self.query_rgb(rgb)
        rgb_k = self.key_rgb(rgb)
        rgb_v = self.value_rgb(rgb)

        d_q = self.query_d(d)
        d_k = self.key_d(d)
        d_v = self.value_d(d)

        t_q = self.query_t(t)
        t_k = self.key_t(t)
        t_v = self.value_t(t)

        d_q1 = d_q.view(batchsize, -1, height*width).permute(0, 2, 1)
        t_k1 = t_k.view(batchsize, -1, height*width)
        energy1 = torch.bmm(d_q1, t_k1)
        attention1 = self.softmax(energy1)
        rgb_v1 = rgb_v.view(batchsize, -1, height*width)
        out1 = torch.bmm(rgb_v1, attention1.permute(0, 2, 1))
        out1 = out1.view(batchsize, C, height, width)
        out1 = self.gamma_1 * out1 + out1

        t_q2 = t_q.view(batchsize, -1, height*width).permute(0, 2, 1)
        rgb_k2 = rgb_k.view(batchsize, -1, height*width)
        energy2 = torch.bmm(t_q2, rgb_k2)
        attention2 = self.softmax(energy2)
        d_v2 = d_v.view(batchsize, -1, height*width)
        out2 = torch.bmm(d_v2, attention2.permute(0, 2, 1))
        out2 = out2.view(batchsize, C, height, width)
        out2 = self.gamma_2 * out2 + out2

        rgb_q3 = rgb_q.view(batchsize, -1, height*width).permute(0, 2, 1)
        d_k3 = d_k.view(batchsize, -1, height*width)
        energy3 = torch.bmm(rgb_q3, d_k3)
        attention3 = self.softmax(energy3)
        t_v3 = t_v.view(batchsize, -1, height*width)
        out3 = torch.bmm(t_v3, attention3.permute(0, 2, 1))
        out3 = out3.view(batchsize, C, height, width)
        out3 = self.gamma_3 * out3 + out3

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv1(out)

        d5 = torch.cat([f5 * out, f5 + out], dim=1)
        d5 = self.conv2(d5)

        return d5
