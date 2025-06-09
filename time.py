import torch
from thop import profile
import numpy as np
from models.Swin_Transformer import SwinNet
# import torchstat as ts
import time
from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    #net1 = SwinNet().cuda()
    net = SwinNet()

    #input1 = [torch.randn((1, 3, 384, 384)) for i in range(100)]
    #input2 = [torch.randn((1, 3, 384, 384)) for i in range(100)]
    #res, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig, edge = net(input1,input2)
    # R1 = transforms.ToPILImage()(R1.squeeze(0))
    # I1 = transforms.ToPILImage()(I1.squeeze(0))
    # flops, params = profile(net1, inputs=(input,  ))
    input1 = torch.randn(1, 3, 384, 384)
    input2 = torch.randn(1, 3, 384, 384)
    flops, params = profile(net, inputs=(input1, input2))

    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')




