import numpy as np
import logging
import os
from metric_data import test_dataset
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm

datasets = ['D-challenge', 'V-challenge', 'T-challenge']
save_path = './model/'
logging.basicConfig(filename=save_path + 'metric_score-HWSI.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')

for dataset in datasets:
    lines = os.listdir(os.path.join('../challenge in Test-set/', dataset))
    for line in lines:
        dataset_path = os.path.join('../challenge in Test-set/', dataset, line) ##gt_path

        dataset_path_pre = os.path.join('./challenge-HWSI/', dataset, line)  ##pre_salmap_path
        # sal_root = os.path.join(dataset_path_pre, dataset)
        sal_root = dataset_path_pre
        gt_root = dataset_path +'/GT/'
        test_loader = test_dataset(sal_root, gt_root)
        mae,fm,sm,em,wfm= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm()
        for i in range(test_loader.size):
            print ('predicting for %d / %d' % ( i + 1, test_loader.size))
            sal, gt = test_loader.load_data()
            if sal.size != gt.size:
                x, y = gt.size
                sal = sal.resize((x, y))
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            gt[gt > 0.5] = 1
            gt[gt != 1] = 0
            res = sal
            res = np.array(res)
            if res.max() == res.min():
                res = res/255
            else:
                res = (res - res.min()) / (res.max() - res.min())
            mae.update(res, gt)
            sm.update(res,gt)
            fm.update(res, gt)
            em.update(res,gt)
            wfm.update(res,gt)

        MAE = mae.show()
        maxf,meanf,_,_ = fm.show()
        sm = sm.show()
        em = em.show()
        wfm = wfm.show()
        print('dataset: {} MAE: {:.8f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Em: {:.4f} Sm: {:.4f}'.format(dataset, MAE, maxf,meanf,wfm,em,sm))
        logging.info(
            'dataset: {} lines:{} MAE: {:.8f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Em: {:.4f} Sm: {:.4f} '.
            format(dataset, line, MAE, maxf,meanf,wfm,em,sm))