import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from net import Mnet
import numpy as np
from swinoptions import opt
from data_edge import VDT_test_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    test_image_root = opt.test_rgb_root
    test_gt_root = opt.test_gt_root
    test_depth_root = opt.test_depth_root
    test_t_root = opt.test_t_root
    test_loader1 = VDT_test_dataset(test_image_root, test_gt_root, test_depth_root, test_t_root, opt.trainsize)
    model_path = 'model/final.pth'
    out_path = 'output222/'
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
        for i in range(test_loader1.size):
            image, gt, d, t, name, img_for_post = test_loader1.load_data()
            print(name)
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            d = d.repeat(1, 3, 1, 1).cuda()
            t = t.repeat(1, 3, 1, 1).cuda()
            res, score2, score3, score4, score5, s_sig, s2_sig, s3_sig, s4_sig, s5_sig = net(image, t, d)
            score = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            pred = np.squeeze(torch.sigmoid(score).cpu().data.numpy())
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            cv2.imwrite(os.path.join(out_path + name), 255 * pred)
            print(os.path.join(out_path + name))

    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))



