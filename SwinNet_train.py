# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: SwinNet.py
@time: 2021/5/6 16:12
"""
import os
import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')

import numpy as np
from datetime import datetime
from VDTNet import Mnet
from torchvision.utils import make_grid
from data_edge import get_loader, VDT_test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from swinoptions import opt
import yaml
import pytorch_iou


# set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
t_root = opt.t_root

test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
test_t_root = opt.test_t_root

save_path = opt.save_path

logging.basicConfig(filename=save_path + 'VDTNet.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')


# set yaml config path
# yaml_path = './config/module.yaml'
# config = yaml.load(open(yaml_path, 'r'), yaml.SafeLoader)

# build the model
# model = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
# model = Mnet()

num_parms = 0
# if (opt.load is not None):
#     model.load_pretrained_model()
#     # model.load_state_dict(torch.load(opt.load)['model'], strict=False)
#     # model.load_state_dict(torch.load(opt.load))
#     # model.load_state_dict(torch.load(opt.load_pre))
#     print('load model from ', opt.load)

model = Mnet().cuda()
model.load_pretrained_model()
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, t_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = VDT_test_dataset(test_image_root, test_gt_root, test_depth_root, test_t_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
ECE = torch.nn.BCELoss()
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0

IOU = pytorch_iou.IOU(size_average=True)

# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    sal_loss_all = 0
    edge_loss_all = 0
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, depth, t) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            rgb = images.cuda()
            gts = gts.cuda()
            d = depth.repeat(1,3,1,1).cuda()
            t = t.repeat(1,3,1,1).ocuda()
            s, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig, = model(rgb, t, d)
            sal_loss1 = F.binary_cross_entropy_with_logits(s1, gts, reduction='mean')
            sal_loss2 = F.binary_cross_entropy_with_logits(s2, gts, reduction='mean')
            sal_loss3 = F.binary_cross_entropy_with_logits(s3, gts, reduction='mean')
            # sml = get_saliency_smoothness(torch.sigmoid(s), label)
            sal_loss = F.binary_cross_entropy_with_logits(s, gts, reduction='mean')
            loss1 = sal_loss + IOU(s_sig, gts)
            loss2 = sal_loss1 + IOU(s1_sig, gts)
            loss3 = sal_loss2 + IOU(s2_sig, gts)
            loss4 = sal_loss3 + IOU(s3_sig, gts)

            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            # sal_loss_all += sal_loss.data
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:.4f}  loss4:{:.4f} '.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data,
                           loss4.data, ))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:.4f} Loss3: {:.4f}  loss4:{:.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data, loss4.data,
                           ))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')
        # sal_loss_all /= epoch_step
        loss_all /= epoch_step
        print('Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'SwinTransNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'SwinTransNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, t, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            rgb = image.cuda()
            d = depth.repeat(1,3,1,1).cuda()
            t = t.repeat(1, 3, 1, 1).cuda()
            res, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig, = model(rgb, t, d)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'SwinTransNet_epoch_best2.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    for epoch in range(1, opt.epoch):
        # if (epoch % 50 ==0 and epoch < 60):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)

        test(test_loader, model, epoch, save_path)
