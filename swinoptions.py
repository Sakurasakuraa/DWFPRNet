# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: options.py
@time: 2021/5/16 14:52

"""
import argparse
# RGBD
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='./vgg16.pth', help='train from checkpoints')
#parser.add_argument('--load_pre', type=str, default='./SwinTransNet_RGBD_cpts/SwinTransNet_epoch_best.pth', help='train from checkpoints')
parser.add_argument('--load_pre', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='../VDT-2048/Train/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='../VDT-2048/Train/D/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='../VDT-2048/Train/GT/', help='the training gt images root')
parser.add_argument('--t_root', type=str, default='../VDT-2048/Train/T/', help='the training t images root')
parser.add_argument('--test_rgb_root', type=str, default='../VDT-2048/Test/RGB/', help='the test rgb images root')
parser.add_argument('--test_depth_root', type=str, default='../VDT-2048/Test/D/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='../VDT-2048/Test/GT/', help='the test gt images root')
parser.add_argument('--test_t_root', type=str, default='../VDT-2048/Test/T/', help='the test t images root')

parser.add_argument('--save_path', type=str, default='./cpts/', help='the path to save models and logs')
opt = parser.parse_args()
