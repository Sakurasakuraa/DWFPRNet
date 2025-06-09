coding = 'utf-8'
import logging
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
IOU = pytorch_iou.IOU(size_average = True)

def my_loss1(s, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig, label):
    sal_loss1 = F.binary_cross_entropy_with_logits(s1, label, reduction='mean')
    sal_loss2 = F.binary_cross_entropy_with_logits(s2, label, reduction='mean')
    sal_loss3 = F.binary_cross_entropy_with_logits(s3, label, reduction='mean')
    # sml = get_saliency_smoothness(torch.sigmoid(s), label)
    sal_loss = F.binary_cross_entropy_with_logits(s, label, reduction='mean')
    loss1 = sal_loss + IOU(s_sig, label)
    loss2 = sal_loss1 + IOU(s1_sig, label)
    loss3 = sal_loss2 + IOU(s2_sig, label)
    loss4 = sal_loss3 + IOU(s3_sig, label)
    return loss1 + loss2 + loss3 + loss4


# random.seed(825)
# np.random.seed(825)
# torch.manual_seed(825)
# torch.cuda.manual_seed(825)
# torch.cuda.manual_seed_all(825)

# dataset
train_img_root = '../VDT-2048/Train/'
test_img_root = '../VDT-2048/Test/'
save_path = './model'

logging.basicConfig(filename=save_path + 'VDTNet.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
# model_path = './model/epoch_20.pth'
if not os.path.exists(save_path): os.mkdir(save_path)
lr = 0.001
batch_size = 4
epoch = 100
# lr_dec = [18, 34, 43, 53]
lr_dec = [30, 60, 90]
num_params = 0

net = Mnet()
net.load_pretrained_model()
net.cuda()
# print('loading model from %s...' % model_path)
# net.load_state_dict(torch.load(model_path))
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr)
for p in net.parameters():
    num_params += p.numel()
print("The number of parameters: {}".format(num_params))

print('load data...')
train_data = Data(train_img_root)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers = 8)

test_data = Data(root='../VDT-2048/Test/', mode='test')
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

iter_num = len(train_loader)

def train(train_loader, net, optimizer, epochi, save_path):
    net.train()

    # for epochi in tqdm(range(1, epoch + 1)):
    # if epochi in lr_dec:
    #     lr = lr / 10
    #     # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,
    #     #                       momentum=0.9)
    #     optimizer = torch.optim.Adam(net.parameters(), lr)
    #     print(lr)
    prefetcher = DataPrefetcher(train_loader)
    rgb, t, d, label = prefetcher.next()
    r_sal_loss = 0
    epoch_ave_loss = 0
    net.zero_grad()
    i = 0
    while rgb is not None:
        i += 1
        s, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig, = net(rgb, t, d)
        sal_loss1 = F.binary_cross_entropy_with_logits(s1, label, reduction='mean')
        sal_loss2 = F.binary_cross_entropy_with_logits(s2, label, reduction='mean')
        sal_loss3 = F.binary_cross_entropy_with_logits(s3, label, reduction='mean')
        # sml = get_saliency_smoothness(torch.sigmoid(s), label)
        sal_loss = F.binary_cross_entropy_with_logits(s, label, reduction='mean')
        loss1 = sal_loss + IOU(s_sig, label)
        loss2 = sal_loss1 + IOU(s1_sig, label)
        loss3 = sal_loss2 + IOU(s2_sig, label)
        loss4 = sal_loss3 + IOU(s3_sig, label)
        sal_loss = loss1 + loss2 + loss3 + loss4
        r_sal_loss += sal_loss.data
        sal_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f, lr: %7.6f' % (
                epochi, epoch, i, iter_num, r_sal_loss / 100, lr,))
            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:.4f} Loss3: {:.4f}  loss4:{:.4f} '.
                format(epoch, epochi, i, iter_num, loss1.data, loss2.data, loss3.data, loss4.data,
                       ))
            epoch_ave_loss += (r_sal_loss / 100)
            r_sal_loss = 0
        rgb, t, d, label = prefetcher.next()
    print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_ave_loss / (10.5 / batch_size))))
    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epochi, epoch, (epoch_ave_loss / (10.5 / batch_size))))
    if (epochi <= 50) and (epochi % 10 == 0):
        torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
    elif (epochi > 50) and (epochi % 5 == 0):
        torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))

def test(test_loader, net, epoch, save_path):
    global best_mae, best_epoch
    net.eval()
    with torch.no_grad():
        mae_sum = 0
        for rgb, t, d, gt, (H, W), name in test_loader:
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            res, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig, = net(rgb.cuda().float(), t.cuda().float(), d.cuda().float())
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(net.state_dict(), save_path + 'VDTNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    for epoch in range(1, epoch):
        # if (epoch % 50 ==0 and epoch < 60):
        train(train_loader, net, optimizer, epoch, save_path)

        test(test_loader, net, epoch, save_path)