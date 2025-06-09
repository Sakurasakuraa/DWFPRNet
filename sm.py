import numpy as np
import logging
import os
from metric_data import test_dataset
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm,Eval_Smeasure


dataset_path = '../VDT-2048/Test/' ##gt_path

dataset_path_pre = '../VDT_origin/'  ##pre_salmap_path

# test_datasets = ['VDT10.15_4']     ##test_datasets_name
test_datasets = ['final_result']
for dataset in test_datasets:
    sal_root = os.path.join(dataset_path_pre, dataset)
    # sal_root = os.path.join(dataset)
    gt_root = dataset_path +'/GT/'
    test_loader = test_dataset(sal_root, gt_root)
    # mae,fm,sm,em,wfm= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm()
    mae, fm, em, wfm = cal_mae(), cal_fm(test_loader.size), cal_em(), cal_wfm()

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
        # sm.update(res,gt)
        s = Eval_Smeasure()
        fm.update(res, gt)
        em.update(res,gt)
        wfm.update(res,gt)

    MAE = mae.show()
    maxf,meanf,_,_ = fm.show()
    # sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    print('dataset: {} MAE: {:.8f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Em: {:.4f} Sm: {:.4f}'.format(dataset, MAE, maxf,meanf,wfm,em,s))

