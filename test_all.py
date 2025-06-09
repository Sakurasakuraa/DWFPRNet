import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from VDTNet import Mnet
import numpy as np

import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    # model_path = './model/VDTNet_loss_best.pth'
    model_path = './model/VDTNet_epoch_best12.25_dbrige.pth'
    out_path = './VDT12.25-dbrige/'
    data = Data(root='../VDT-2048/Test/', mode='test')
    loader = DataLoader(data, batch_size=1, shuffle=False)
    net = Mnet().cuda()
    print('loading model from %s...' % model_path)
    net.load_state_dict(torch.load(model_path))
    if not os.path.exists(out_path): os.mkdir(out_path)
    time_s = time.time()
    img_num = len(loader)
    net.eval()
    with torch.no_grad():
        for rgb, t, d, gt, (H, W), name in loader:
            print(name[0])
            # score, score2, score3, score4, score5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = net(rgb.cuda().float(), t.cuda().float(), d.cuda().float())
            score, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig = net(rgb.cuda().float(), t.cuda().float(), d.cuda().float())
            score = F.interpolate(score, size=(H, W), mode='bilinear',align_corners=True)
            # gt = F.interpolate(gt, size=(H, W), mode='bilinear',align_corners=True)
            pred = np.squeeze(torch.sigmoid(score).cpu().data.numpy())
            # g_pred = np.squeeze(torch.sigmoid(gt).cpu().data.numpy())
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            # plt.imshow(g_pred)
            # plt.show()
            cv2.imwrite(os.path.join(out_path, name[0][3:-4] + '.png'), 255 * pred)
    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))



