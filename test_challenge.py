import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from VDTNet import Mnet
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    model_path = './model/VDTNet_epoch_best12.25_3.pth'
    datasets = [ 'D-challenge','V-challenge', 'T-challenge']
    for dataset in datasets:
        lines = os.listdir(os.path.join('../challenge in Test-set/', dataset))
        for line in lines:
            data = Data(root=os.path.join('../challenge in Test-set/', dataset, line), mode='test')
            out_path = os.path.join('./challenge/', dataset, line)
            # if not os.path.exists(out_path): os.mkdir(out_path)
            loader = DataLoader(data, batch_size=1, shuffle=False)
            net = Mnet().cuda()
            print('loading model from %s...' % model_path)
            net.load_state_dict(torch.load(model_path))
            if not os.path.exists(out_path): os.mkdir(out_path)
            time_s = time.time()
            img_num = len(loader)
            net.eval()
            with torch.no_grad():
                for rgb, t, d, _, (H, W), name in loader:
                    print(name[0])
                    score, s1, s2, s3, s_sig, s1_sig, s2_sig, s3_sig = net(rgb.cuda().float(), t.cuda().float(), d.cuda().float())
                    score = F.interpolate(score, size=(H, W), mode='bilinear',align_corners=True)
                    pred = np.squeeze(torch.sigmoid(score).cpu().data.numpy())
                    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    cv2.imwrite(os.path.join(out_path, name[0][20:-4] + '.png'), 255 * pred)
            time_e = time.time()
            print('speed: %f FPS' % (img_num / (time_e - time_s)))



