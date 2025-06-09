import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from VDTNet import Mnet
import numpy as np
from swinoptions import opt
from data_edge import VDT_test_dataset
from metric_data import test_dataset
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm

import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
test_t_root = opt.test_t_root



if __name__ == '__main__':
    # model_path = './model/VDTNet_loss_best.pth'
    model_path = './model/VDTNet_loss_best150.pth'
    out_path = './output/817test/'
    data = Data(root='../VDT-2048/Test/', mode='test')
    loader = DataLoader(data, batch_size=1, shuffle=False)
    net = Mnet().cuda()
    print('loading model from %s...' % model_path)
    net.load_state_dict(torch.load(model_path))
    if not os.path.exists(out_path): os.mkdir(out_path)
    time_s = time.time()
    img_num = len(loader)
    net.eval()
    mae_sum = 0
    mae = cal_mae
    dataset_path = '../VDT-2048/Test/'  ##gt_path

    sal_root = '../VDT-2048/Test/GT/'  ##pre_salmap_path

    # test_datasets = ['test0']  ##test_datasets_name
    with torch.no_grad():
        gt_root = dataset_path + '/GT/'
        test_loader = test_dataset(sal_root, gt_root)
        i = 0
        for rgb, t, d, _, (H, W), name in loader:
            print(name[0])
            _, gt = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            gt[gt > 0.5] = 1
            gt[gt != 1] = 0
            score, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig = net(rgb.cuda().float(), t.cuda().float(), d.cuda().float())
            score = F.interpolate(score, size=(H, W), mode='bilinear',align_corners=True)
            pred = np.squeeze(torch.sigmoid(score).cpu().data.numpy())
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            mae_sum += np.sum(np.abs(pred - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            # plt.imshow(g_pred)
            len = test_loader.size
            # plt.show()
            cv2.imwrite(os.path.join(out_path, name[0][3:-4] + '.png'), 255 * pred)
            i = i+1
            mae = mae_sum / test_loader.size

    time_e = time.time()
    print("MAE:{}".format(mae))
    print('speed: %f FPS' % (img_num / (time_e - time_s)))


