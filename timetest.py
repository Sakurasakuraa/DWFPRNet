import torch
from thop import profile
import  os
# from model.BBSNet_model import BBSNet
from VDTNet import Mnet
# import torchstat as ts
import time
from PIL import Image
from torchvision import transforms


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

if __name__ == '__main__':
    net = Mnet().cuda()
    net.eval()
    input1 = torch.randn((1, 3, 352, 352)).cuda()
    input2 = torch.randn((1, 3, 352, 352)).cuda()
    input3 = torch.randn((1, 3, 352, 352)).cuda()
    for i in range(10):  # 预热
        score, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig = net(input1, input2, input3)
    T1 = 0
    print('开始计算时间')
    for i in range(100):
        torch.cuda.synchronize()
        t1 = time.time()
        score, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig = net(input1, input2, input3)
        torch.cuda.synchronize()
        t2 = time.time()
        T1 += t2 - t1

    input1 = torch.randn((1, 3, 352, 352)).cuda()
    input2 = torch.randn((1, 3, 352, 352)).cuda()
    input3 = torch.randn((1, 3, 352, 352)).cuda()
    flops, params = profile(net, inputs=(input1, input2, input3))

    modelsize = getModelSize(net)

    print("FPS: %f" % (1.0 / (T1 / 100)))
    print('平均推理时间')
    print(T1 / 100)  # 单位为秒
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

