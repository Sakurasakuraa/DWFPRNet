import torch
from torch import nn
import torch.nn.functional as F
from modules.aspp815 import aspp
from modules.fuse914 import fuse
from modules.quality815 import quality
from modules.decoder import decoder
import vgg

def convblock(in_, out_, ks, st=1, pad=0,dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias = False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        self.rgb_net = vgg.a_vgg16()
        self.t_net = vgg.a_vgg16()
        self.d_net = vgg.a_vgg16()

        self.vd1 = fuse(64)
        self.vd2 = fuse(128)
        self.vd3 = fuse(256)
        self.vd4 = fuse(512)
        self.vd5 = fuse(512)

        self.vt1 = fuse(64)
        self.vt2 = fuse(128)
        self.vt3 = fuse(256)
        self.vt4 = fuse(512)
        self.vt5 = fuse(512)

        self.fuse1 = quality(64)
        self.fuse2 = quality(128)
        self.fuse3 = quality(256)
        self.fuse4 = quality(512)
        self.fuse5 = quality(512)

        # self.attention = attention(512)

        self.decoder = decoder()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb, t, d):
        d = d.repeat(1, 3, 1, 1)
        t = t.repeat(1, 3, 1, 1)
        rgb_f = self.rgb_net(rgb)
        d_f = self.d_net(d)
        t_f = self.t_net(t)

        feature_list = []

        vd1 = self.vd1(rgb_f[0], d_f[0])
        vt1 = self.vt1(rgb_f[0], t_f[0])
        vd2 = self.vd2(rgb_f[1], d_f[1])
        vt2 = self.vt2(rgb_f[1], t_f[1])
        vd3 = self.vd3(rgb_f[2], d_f[2])
        vt3 = self.vt3(rgb_f[2], t_f[2])
        vd4 = self.vd4(rgb_f[3], d_f[3])
        vt4 = self.vt4(rgb_f[3], t_f[3])
        vd5 = self.vd5(rgb_f[4], d_f[4])
        vt5 = self.vt5(rgb_f[4], t_f[4])




        f1, rd1_w1, rt1_w1, rd1_w2, rt1_w2, t1_w, d1_w = self.fuse1(vd1, vt1, d_f[0], t_f[0])
        f2, rd2_w1, rt2_w1, rd2_w2, rt2_w2, t2_w, d2_w = self.fuse2(vd2, vt2, d_f[1], t_f[1])
        f3, rd3_w1, rt3_w1, rd3_w2, rt3_w2, t3_w, d3_w = self.fuse3(vd3, vt3, d_f[2], t_f[2])
        f4, rd4_w1, rt4_w1, rd4_w2, rt4_w2, t4_w, d4_w = self.fuse4(vd4, vt4, d_f[3], t_f[3])
        f5, rd5_w1, rt5_w1, rd5_w2, rt5_w2, t5_w, d5_w = self.fuse5(vd5, vt5, d_f[4], t_f[4])

        feature_list.append(f1)
        feature_list.append(f2)
        feature_list.append(f3)
        feature_list.append(f4)
        feature_list.append(f5)


        # attention = self.attention(rgb_e5, d_e5, t_e5, f5)

        s, s1, s2, s3 = self.decoder(feature_list)
        return vd1,vt1,vd2,vt2,vd3,vt3,vd4,vt4,vd5,vt5,f1,f2,f3,f4,f5
        # return rd1_w1,rt1_w1,rd1_w2,rt1_w2,t1_w,d1_w,\
        #     rd2_w1,rt2_w1,rd2_w2,rt2_w2,t2_w,d2_w,\
        #     rd3_w1,rt3_w1,rd3_w2,rt3_w2,t3_w,d3_w,\
        #     rd4_w1,rt4_w1,rd4_w2,rt4_w2,t4_w,d4_w,\
        #     rd5_w1,rt5_w1,rd5_w2,rt5_w2,t5_w,d5_w

    def load_pretrained_model(self):
        st=torch.load("vgg16.pth")
        st2={}
        for key in st.keys():
            st2['base.'+key]=st[key]
        self.rgb_net.load_state_dict(st2)
        self.t_net.load_state_dict(st2)
        self.d_net.load_state_dict(st2)
        print('loading pretrained model success!')