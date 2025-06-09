import torch
from torch import nn
import torch.nn.functional as F
import vgg
from torch.nn import Conv2d, Parameter, Softmax

def convblock(in_, out_, ks, st=1, pad=0,dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias = False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class CASA(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(CASA, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        x = x.mul(self.sigmoid(out))

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class MBF(nn.Module):
    def __init__(self, in_):
        super(MBF, self).__init__()
        self.relu = nn.ReLU(True)
        self.rgb_1 = convblock(in_, in_ // 4, 3, 1, 1, 1)
        self.rgb_2 = convblock(in_, in_ // 4, 3, 1, 2, 2)
        self.rgb_3 = convblock(in_, in_ // 4, 3, 1, 4, 4)
        self.rgb_4 = convblock(in_, in_ // 4, 3, 1, 6, 6)

        self.dt_1 = convblock(in_, in_ // 4, 3, 1, 1, 1)
        self.dt_2 = convblock(in_, in_ // 4, 3, 1, 2, 2)
        self.dt_3 = convblock(in_, in_ // 4, 3, 1, 4, 4)
        self.dt_4 = convblock(in_, in_ // 4, 3, 1, 6, 6)

        self.rgb_1_casa = CASA(in_ // 4, 3)
        self.rgb_2_casa = CASA(in_ // 4, 3)
        self.rgb_3_casa = CASA(in_ // 4, 3)
        self.rgb_4_casa = CASA(in_ // 4 ,3)

        self.dt_1_casa = CASA(in_ // 4, 3)
        self.dt_2_casa = CASA(in_ // 4, 3)
        self.dt_3_casa = CASA(in_ // 4, 3)
        self.dt_4_casa = CASA(in_ // 4, 3)

        self.casa = CASA(in_)
        self.conv = convblock(2*in_, in_, 3, 1, 1, 1)

    def forward(self, x_rgb, x_dt):
        x1_rgb = self.rgb_1(x_rgb)
        x2_rgb = self.rgb_2(x_rgb)
        x3_rgb = self.rgb_3(x_rgb)
        x4_rgb = self.rgb_4(x_rgb)

        x1_dt = self.dt_1(x_dt)
        x2_dt = self.dt_2(x_dt)
        x3_dt = self.dt_3(x_dt)
        x4_dt = self.dt_4(x_dt)

        x1_dt_sa = x1_dt.mul(self.rgb_1_casa(x1_rgb))
        x1_rgb_sa = x1_rgb.mul(self.dt_1_casa(x1_dt))
        x1_dt = x1_dt + x1_dt_sa
        x1_rgb = x1_rgb + x1_rgb_sa
        x1 = torch.cat((x1_dt, x1_rgb),1)

        x2_dt_sa = x2_dt.mul(self.rgb_2_casa(x2_rgb))
        x2_rgb_sa = x2_rgb.mul(self.dt_2_casa(x2_dt))
        x2_dt = x2_dt + x2_dt_sa
        x2_rgb = x2_rgb + x2_rgb_sa
        x2 = torch.cat((x2_dt, x2_rgb), 1)

        x3_dt_sa = x3_dt.mul(self.rgb_3_casa(x3_rgb))
        x3_rgb_sa = x3_rgb.mul(self.dt_3_casa(x3_dt))
        x3_dt = x3_dt + x3_dt_sa
        x3_rgb = x3_rgb + x3_rgb_sa
        x3 = torch.cat((x3_dt, x3_rgb), 1)

        x4_dt_sa = x4_dt.mul(self.rgb_4_casa(x4_rgb))
        x4_rgb_sa = x4_rgb.mul(self.dt_4_casa(x4_dt))
        x4_dt = x4_dt + x4_dt_sa
        x4_rgb = x4_rgb + x4_rgb_sa
        x4 = torch.cat((x4_dt, x4_rgb), 1)

        y = self.conv(torch.cat((x1, x2, x3, x4), 1))
        z = y.mul(self.casa(y)) + x_rgb.mul(self.casa(y)) + x_dt.mul(self.casa(y))

        return z

class GFAPF(nn.Module):
    def __init__(self):
        super(GFAPF, self).__init__()
        self.de_chan = convblock(1536, 256, 3, 1, 1, 1)
        self.convd1 = convblock(256, 128, 3, 1, 1, 1)
        self.convd2 = convblock(256, 128, 3, 1, 2, 2)
        self.convd3 = convblock(256, 128, 3, 1, 4, 4)
        self.convd4 = convblock(256, 128, 3, 1, 6, 6)
        self.convd5 = convblock(256, 128, 1, 1, 0, 1)
        self.fus = convblock(640, 512, 1, 1, 0, 1)
        self.casa1 = CASA(512)
        self.casa = CASA(128)

    def forward(self, rgb, t, d):
        rgb = rgb.mul(self.casa1(d)) + rgb.mul(self.casa1(t))
        t = t.mul(self.casa1(d)) + t.mul(self.casa1(rgb))
        d = d.mul(self.casa1(rgb)) + d.mul(self.casa1(t))
        rgbtd = self.de_chan(torch.cat((rgb, t, d), 1))
        out1 = self.convd1(rgbtd)
        rgbtd1 = rgbtd.mul(self.casa(out1))
        out2 = self.convd2(rgbtd1)
        rgbtd2 = rgbtd1.mul(self.casa(out2))
        out3 = self.convd3(rgbtd2)
        rgbtd3 = rgbtd2.mul(self.casa(out3))
        out4 = self.convd4(rgbtd3)
        rgbtd4 = rgbtd3.mul(self.casa(out4))
        out5 = F.interpolate(self.convd5(F.adaptive_avg_pool2d(rgbtd4, 2)), rgb.size()[2:], mode='bilinear', align_corners=True)
        out = self.fus(torch.cat((out1, out2, out3, out4, out5), 1))
        # out2 = self.fus2(torch.cat((rgbtd1, rgbtd2, rgbtd3, rgbtd4, rgbtd), 1))
        return out

class GFB(nn.Module):
    def __init__(self, in_1, in_2):
        super(GFB, self).__init__()
        self.s_mask=nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(1,1,3,1,1),
            nn.Sigmoid()
        )
        self.conv_globalinfo = convblock(in_2, 128, 3, 1, 1)
        self.conv_rt = convblock(in_1, 128, 3, 1, 1)
        self.conv_tr = convblock(in_1, 128, 3, 1, 1)
        self.conv_out = convblock(128, in_1, 3, 1, 1)

    def forward(self, cur1, cur2, global_info):
        cur_size = cur1.size()[2:]
        global_info = self.conv_globalinfo(F.interpolate(global_info, cur_size, mode='bilinear',align_corners=True))

        att1_1 = torch.unsqueeze(torch.max(cur2,1)[0], 1)
        att1_2 = torch.unsqueeze(torch.mean(cur2, 1), 1)
        atten1 = self.s_mask(torch.cat((att1_1,att1_2), 1))
        fus_rt = self.conv_rt((cur1 + torch.mul(atten1, cur1)))

        att2_1 = torch.unsqueeze(torch.max(cur1,1)[0], 1)
        att2_2 = torch.unsqueeze(torch.mean(cur1, 1), 1)
        atten2 = self.s_mask(torch.cat((att2_1,att2_2), 1))
        fus_tr = self.conv_tr((cur2 + torch.mul(atten2, cur2)))

        fus = fus_rt + fus_tr + global_info
        return self.conv_out(fus)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.glo = GFAPF()
        self.S5 = nn.Conv2d(512, 1, 3, stride=1, padding=1)
        self.S4 = nn.Conv2d(512, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(128, 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.mbf5_rgbd = MBF(512)
        self.mbf4_rgbd = MBF(512)
        self.mbf3_rgbd = MBF(256)
        self.mbf2_rgbd = MBF(128)
        self.mbf1_rgbd = MBF(64)

        self.gfb5 = GFB(512, 512)
        self.gfb4 = GFB(512, 512)
        self.gfb3 = GFB(256, 512)
        self.gfb2 = GFB(128, 256)
        self.gfb1 = GFB(64, 128)

        self.mbf5_td = MBF(512)
        self.mbf4_td = MBF(512)
        self.mbf3_td = MBF(256)
        self.mbf2_td = MBF(128)
        self.mbf1_td = MBF(64)

    def forward(self, rgb, t, d):
        xsize = rgb[0].size()[2:]
        rgbdt_g = self.glo(rgb[4], t[4], d[4])

        rgbd5 = self.mbf5_rgbd(rgb[4], d[4])
        td5 = self.mbf5_td(t[4], d[4])
        rgbdt5 = self.gfb5(rgbd5, td5, rgbdt_g)
        s5 = self.S5(rgbdt5)

        rgbd4 = self.mbf4_rgbd(rgb[3], d[3])
        td4 = self.mbf4_td(t[3], d[3])
        rgbdt4 = self.gfb4(rgbd4, td4, rgbdt5)
        s4 = self.S4(rgbdt4)

        rgbd3 = self.mbf3_rgbd(rgb[2], d[2])
        td3 = self.mbf3_td(t[2], d[2])
        rgbdt3 = self.gfb3(rgbd3, td3, rgbdt4)
        s3 = self.S3(rgbdt3)

        rgbd2 = self.mbf2_rgbd(rgb[1], d[1])
        td2 = self.mbf2_td(t[1], d[1])
        rgbdt2 = self.gfb2(rgbd2, td2, rgbdt3)
        s2 = self.S2(rgbdt2)

        rgbd1 = self.mbf1_rgbd(rgb[0], d[0])
        td1 = self.mbf1_td(t[0], d[0])
        rgbdt1 = self.gfb1(rgbd1, td1, rgbdt2)
        s1 = self.S1(rgbdt1)

        s5 = F.interpolate(s5, xsize, mode='bilinear', align_corners=True)
        s4 = F.interpolate(s4, xsize, mode='bilinear', align_corners=True)
        s3 = F.interpolate(s3, xsize, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, xsize, mode='bilinear', align_corners=True)
        return s1, s2, s3, s4, s5

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        self.rgb_net = vgg.a_vgg16()
        self.t_net = vgg.a_vgg16()
        self.d_net = vgg.a_vgg16()
        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb, t, d):
        rgb_f = self.rgb_net(rgb)
        t_f = self.t_net(t)
        d_f = self.d_net(d)
        score1, score2, score3, score4, score5 = self.decoder(rgb_f, t_f, d_f)
        return score1, score2, score3, score4, score5, self.sigmoid(score1), self.sigmoid(score2), self.sigmoid(score3), self.sigmoid(score4), self.sigmoid(score5)

    def load_pretrained_model(self):
        st=torch.load("vgg16.pth")
        st2={}
        for key in st.keys():
            st2['base.'+key]=st[key]
        self.rgb_net.load_state_dict(st2)
        self.t_net.load_state_dict(st2)
        self.d_net.load_state_dict(st2)
        print('loading pretrained model success!')