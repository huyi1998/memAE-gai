from __future__ import absolute_import, print_function
import os
import scipy.io as sio
import numpy as np
import sklearn.metrics as skmetr
import utils
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def eval_video2(gt_file, score_path, data):
    label_orig = np.load(gt_file, allow_pickle=True)
    if type(score_path) is str:
        score_orig = np.load(score_path)
    else:
        score_orig = score_path
    score_after, label_after = [], []
    init = 0
    for i in range(len(label_orig)):
        _label_use = label_orig[i]
        # 将视频剪辑中的平均重建误差分配给中心帧：即，如果视频剪辑从frame_001.jpg开始并以frame_016.jpg结束，
        # 则假定平均重建误差是帧frame_008.jpg的误差
        # _label_use = _label_use[8:-7]
        _label_use = _label_use[21:]
        _score_use = score_orig[init:init+len(_label_use)]
        init += len(_label_use)
        # 高斯滤波
        # _score_use = gaussian_filter1d(_score_use, sigma=6)
        _score_use = _score_use - np.min(_score_use)
        _score_use = 1 - _score_use / (np.max(_score_use) - np.min(_score_use))
        score_after.append(_score_use)
        label_after.append(1 - _label_use + 1)
    score_after = np.concatenate(score_after, axis=0)
    label_after = np.concatenate(label_after, axis=0)
    print("Number of gt frames:", len(label_after))
    print("Number of predictions:", len(score_after))
    # fpr:False positive rate   tpr:True positive rate   label_after:真实的样本标签   score_after:对每个样本的预测结果   pos_label:正样本的标签
    # threshold返回的结果是score_after内的元素去重后加入一个‘最大值+1’的值降序排序后组成的数据,每一个元素作为阈值，数据类型是一维数组
    fpr, tpr, thresholds = skmetr.roc_curve(label_after, score_after, pos_label=2)
    auc = skmetr.auc(fpr, tpr)

    # EER计算
    # fnr = 1 - tpr
    # # the threshold of fnr == fpr
    # eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    # # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    # eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    # # return the mean of eer from fpr and from fnr
    # eer = (eer_1 + eer_2) / 2
    # print("EER on data %s is %.4f" % (data, eer))

    # 绘制roc曲线
    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    # plt.legend(loc="lower right")
    # plt.show()
    print("AUC score on data %s is %.4f" % (data, auc))


def eval_video(data_path, res_path, is_show=False):
    gt_path = os.path.join(data_path, 'testing_gt/')

    ###
    video_list = utils.get_file_list(gt_path, is_sort=True)
    video_num = len(video_list)

    gt_labels_list = []
    res_prob_list = []
    res_prob_list_org = []

    ###
    for vid_ite in range(video_num):
        gt_file_name = video_list[vid_ite]

        p_idx = [pos for pos, char in enumerate(gt_file_name) if char == '.']
        video_name = gt_file_name[0:p_idx[0]]
        print('Eval: %d/%d-%s' % (vid_ite + 1, video_num, video_name))
        # res file name
        res_file_name = video_name + '.npy'
        # gt file and res file - path
        gt_file_path = os.path.join(gt_path, gt_file_name)
        res_file_path = os.path.join(res_path, res_file_name)
        #     print(gt_file_path)
        #     print(res_file_path)

        # read data
        gt_labels = sio.loadmat(gt_file_path)['l'][0]  # ground truth labels
        res_prob = np.load(res_file_path)  # estimated probability scores
        #     res_prob = np.log10(res_prob)-2*np.log10(255)

        res_prob_list_org = res_prob_list_org + list(res_prob)
        gt_labels_res = gt_labels[8:-7]

        # normalize regularity score
        res_prob_norm = res_prob - res_prob.min()
        res_prob_norm = 1 - res_prob_norm / res_prob_norm.max()

        ##
        gt_labels_list = gt_labels_list + list(1 - gt_labels_res + 1)
        res_prob_list = res_prob_list + list(res_prob_norm)

    fpr, tpr, thresholds = skmetr.roc_curve(np.array(gt_labels_list), np.array(res_prob_list), pos_label=2)
    auc = skmetr.auc(fpr, tpr)
    print(('auc:%f' % auc))

    # output_path = os.path.join(res_path,)
    output_path = res_path
    sio.savemat(os.path.join(output_path, video_name + '_gt_label.mat'),  {'gt_labels_list': np.double(gt_labels_res)}  )
    sio.savemat(os.path.join(output_path, video_name + '_est_label.mat'), {'est_labels_list': np.double(res_prob_list)} )
    acc_file = open(os.path.join(output_path, 'acc.txt'), 'w')
    acc_file.write('{}\nAUC: {}\n'.format(data_path, auc))
    acc_file.close()

    if is_show:
        plt.figure()
        plt.plot(gt_labels_list)
        plt.plot(res_prob_list)

    return auc


if __name__ == "__main__":
    eval_video2("../ckpt/factory3_gt.npy", "../log/factory3/MemAE/lr_0.00020_entropyloss_0.00000_version_12/recons_error_original_1.0_89.npy", "factory3")
