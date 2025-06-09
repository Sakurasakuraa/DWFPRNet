coding = 'utf-8'
import os
from VDTNet import Mnet
from tqdm import tqdm
import torch
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F
from smooth_loss import get_saliency_smoothness
import pytorch_iou
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
IOU = pytorch_iou.IOU(size_average = True)
def my_loss1(score1, score2, score3, score4, score5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig, label):
    sal_loss2 = F.binary_cross_entropy_with_logits(score2, label, reduction='mean')
    sal_loss3 = F.binary_cross_entropy_with_logits(score3, label, reduction='mean')
    sal_loss4 = F.binary_cross_entropy_with_logits(score4, label, reduction='mean')
    sal_loss5 = F.binary_cross_entropy_with_logits(score5, label, reduction='mean')
    sml = get_saliency_smoothness(torch.sigmoid(score1), label)
    sal_loss1 = F.binary_cross_entropy_with_logits(score1, label, reduction='mean')
    loss1 = sal_loss1 + IOU(s1_sig, label)
    loss2 = sal_loss2 + IOU(s2_sig, label)
    loss3 = sal_loss3 + IOU(s3_sig, label)
    loss4 = sal_loss4 + IOU(s4_sig, label)
    loss5 = sal_loss5 + IOU(s5_sig, label)
    return loss1 + loss2 + 0.9*loss3 + 0.8 * sml + 0.8 * loss4 + 0.7 * loss5

if __name__ == '__main__':
    random.seed(825)
    np.random.seed(825)
    torch.manual_seed(825)
    torch.cuda.manual_seed(825)
    torch.cuda.manual_seed_all(825)

    # dataset
    # img_root = '../Train-set/'
    img_root = '../VDT-2048/Train/'
    save_path = './model'
    # model_path = './model/epoch_20.pth'
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.001
    batch_size = 2
    epoch = 60
    lr_dec = [18, 34, 43, 53]
    num_params = 0
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers = 8)
    net = Mnet().cuda()
    net.load_pretrained_model()
    # print('loading model from %s...' % model_path)
    # net.load_state_dict(torch.load(model_path))
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    for p in net.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))
    iter_num = len(loader)
    net.train()

    for epochi in tqdm(range(1, epoch + 1)):
        if epochi in lr_dec:
            lr = lr / 10
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,
                                  momentum=0.9)
            print(lr)
        prefetcher = DataPrefetcher(loader)
        rgb, t, d, label = prefetcher.next()
        r_sal_loss = 0
        epoch_ave_loss = 0
        net.zero_grad()
        i = 0
        while rgb is not None:
            i += 1
            score1, score2, score3, score4, score5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = net(rgb, t, d)
            sal_loss = my_loss1(score1, score2, score3, score4, score5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig,  label)
            r_sal_loss += sal_loss.data
            sal_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f, lr: %7.6f' % (
                    epochi, epoch, i, iter_num, r_sal_loss / 100, lr,))
                epoch_ave_loss += (r_sal_loss / 100)
                r_sal_loss = 0
            rgb, t, d, label = prefetcher.next()
        print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_ave_loss / (10.5 / batch_size))))
        if (epochi <= 45) and (epochi % 10 == 0):
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
        elif (epochi > 45) and (epochi % 2 == 0):
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))



    torch.save(net.state_dict(), '%s/final.pth' % (save_path))