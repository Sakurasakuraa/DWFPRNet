import torch
from thop import profile
from thop import clever_format
# from test import DBCNN
from wodetext import base_resnet
from DecomNet import decomnet
# import torchstat as ts
import time
from PIL import Image
from torchvision import transforms





if __name__ == '__main__':
    net1 = decomnet().cuda()
    net2 = base_resnet().cuda()
    input1 = [torch.randn((1, 3, 936, 1404)).cuda() for i in range(100)]
    T1 = 0
    T2 = 0
    for i in range(2):    #预热
        t1 = time.time()
        R, I = net1(input1[0])
        t2 = time.time()
        t3 = time.time()
        s = net2(R, torch.cat([I, I, I], dim=1))
        t4 = time.time()
        T1 += t2 - t1
        T2 += t4 - t3
        print(T1, T2)
    print('开始计算时间')
    for i in range(100):
        t1 = time.time()
        R,I = net1(input1[i])
        t2 = time.time()
        t3 = time.time()
        s = net2(R,torch.cat([I,I,I],dim=1))
        t4 = time.time()
        T1 += t2 - t1
        T2 += t4 - t3
        print(T1,T2)
    print('平均推理时间')
    print(T1/100)   #单位为秒
    print(T2 / 100)


    input = torch.randn(1, 3, 936, 1404).cuda()
    R1, I1 = net1(input)
    # R1 = transforms.ToPILImage()(R1.squeeze(0))
    # I1 = transforms.ToPILImage()(I1.squeeze(0))
    # flops, params = profile(net1, inputs=(input,  ))
    flops, params = profile(net2, inputs=(R1, torch.cat([I1, I1, I1], dim=1),))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

