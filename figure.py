import matplotlib.pyplot as plt
import sklearn.metrics as skmetr
import numpy as np
from matplotlib.pyplot import MultipleLocator
import torch
from scipy.ndimage import gaussian_filter1d


def show_anomalous_score(gt_file, score_path):
    label_orig = np.load(gt_file, allow_pickle=True)
    if type(score_path) is str:
        score_orig = np.load(score_path)
    else:
        score_orig = score_path
    score_after, label_after = [], []
    init = 0
    for i in range(len(label_orig)):
        _label_use = label_orig[i]
        # _label_use = _label_use[8:-7]
        _label_use = _label_use[21:]
        _score_use = score_orig[init:init + len(_label_use)]
        init += len(_label_use)
        # 高斯滤波
        # _score_use = gaussian_filter1d(_score_use, sigma=6)
        _score_use = _score_use - np.min(_score_use)
        _score_use = 1 - _score_use / np.max(_score_use)   # 公式11
        score_after.append(_score_use)
        label_after.append(1 - _label_use + 1)
    score_conc = np.concatenate(score_after, axis=0)
    label_conc = np.concatenate(label_after, axis=0)
    fpr, tpr, thresholds = skmetr.roc_curve(label_conc, score_conc, pos_label=2)
    auc = skmetr.auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return score_after, label_after, optimal_threshold


def show_tp(score_path, gt_file, vid_seq, dataset):
    """
    Args:
        score_path: str, the path that saves the anomalous score
        gt_file: str, the path that saves the gt label
        vid_seq: the selected video sequence, int
        dataset: str, "UCSD2", "Avenue"
    """
    pred_tot, gt_tot, threshold = show_anomalous_score(gt_file, score_path)
    pred_prob = pred_tot[vid_seq]
    gt_label = gt_tot[vid_seq]
    pred_label = (pred_prob <= threshold).astype('int32')
    tp = [1 for v, j in zip(pred_label, gt_label) if v == 1 and j == 1]
    tpr = np.sum(tp) / len(np.where(gt_label == 1)[0])
    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.add_subplot(111)  # a=1,b=1,c=1,a,b表示把他的行和列都分为1份,c表示分完后的第几个图
    ax.plot([], [], 'b')  # []:x轴数据/y轴数据，'b' 蓝色
    ax.plot([], [], 'r.')
    # ax.plot([], [], 'g')
    # plt.legend(["anomalous score", "anomalous event", "threshold"], fontsize=7, loc='best')
    plt.legend(["normal score", "anomalous event"], fontsize=7, loc='best')
    num_frame = len(pred_prob)
    ax.plot(np.arange(num_frame), pred_prob, 'b')
    ax.plot(np.arange(num_frame)[gt_label == 1], pred_prob[gt_label == 1], 'r.')  # 'r.' 红色+点
    # ax.plot(np.arange(num_frame), [threshold for _ in range(len(gt_label))], 'g:')  # 'g:' 绿色+虚线
    ax.grid(ls=':', alpha=1)  # 网格线
    ax.set_ylabel("Normal score", fontsize=8)
    ax.set_xlabel("Frame", fontsize=8)
    ax.set_title("TPR for sequence %d from dataset %s is: %.2f" % (vid_seq, dataset, tpr * 100),
                 fontsize=8)
    x_major_locator = MultipleLocator(50)
    # 把y轴的刻度间隔设置为10，并存在变量里
    gca = plt.gca()
    # ax为两条坐标轴的实例
    gca.xaxis.set_major_locator(x_major_locator)
    plt.show()

# ---------------------------------------------------------------#
#                        Create Figure                          #
# ---------------------------------------------------------------#


if __name__ == '__main__':
    gt_file_score = "ckpt/factory3_gt.npy"
    recons_score = "log/factory3/ConvAEMem/lr_0.00020_entropyloss_0.00000_version_3/recons_error_original_1.0_85.npy"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for index in 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10:
        show_tp(recons_score, gt_file_score, index, "factory3")