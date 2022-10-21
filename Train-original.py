import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import cv2
import math
from collections import OrderedDict
import copy
import time

import data.utils as data_utils
import models.loss as loss
import utils
from models import AutoEncoderCov3D, AutoEncoderCov3DMem, ConvAEMem

import argparse


def main():
    # print("--------------PyTorch VERSION:", torch.__version__)
    # device = torch.device("cuda:1"if torch.cuda.is_available() else "cpu")
    # print("..............device", device)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    parser = argparse.ArgumentParser(description="MemoryNormality")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs for training')
    parser.add_argument('--val_epoch', type=int, default=2, help='evaluate the model every %d epoch')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--ModelName', help='AE/MemAE', type=str, default='ConvAEMem')
    parser.add_argument('--ModelSetting', help='Conv3D/Conv3DSpar', type=str,
                        default='Conv3DSpar')  # give the layer details later
    parser.add_argument('--MemDim', help='Memory Dimention', type=int, default=2000)
    parser.add_argument('--EntropyLossWeight', help='EntropyLossWeight', type=float, default=0) # 0.0002
    parser.add_argument('--ShrinkThres', help='ShrinkThres', type=float, default=0.0025)
    parser.add_argument('--Suffix', help='Suffix', type=str, default='Non')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='factory3', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='../dataset/', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log/', help='directory of log')
    parser.add_argument('--version', type=int, default=12, help='experiment version')
    parser.add_argument('--resume', type=str, default='log/', help='train the latest model')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue training or not')

    args = parser.parse_args()

    torch.manual_seed(1)

    torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

    # def arrange_image(im_input):
    #     im_input = np.transpose(im_input, (0, 2, 1, 3, 4))
    #     b, t, ch, h, w = im_input.shape
    #     im_input = np.reshape(im_input, [b * t, ch, h, w])
    #     return im_input
    def arrange_image(im_input):
        im_input = np.reshape(im_input, [args.batch_size, -1, args.c, args.h, args.w])
        b, t, ch, h, w = im_input.shape
        im_input = np.reshape(im_input, [b * t, ch, h, w])
        return im_input

    train_folder, test_folder = data_utils.give_data_folder(args.dataset_type,
                                                            args.dataset_path)

    print("The training path", train_folder)
    print("The testing path", test_folder)

    # 图像预处理
    frame_trans = data_utils.give_frame_trans(args.dataset_type, [args.h, args.w])

    train_dataset = data_utils.DataLoader(train_folder, frame_trans, time_step=args.t_length - 1, num_pred=1)
    test_dataset = data_utils.DataLoader(test_folder, frame_trans, time_step=args.t_length - 1, num_pred=1)

    '''
        drop_last：dataset中的数据个数可能不是batch_size的整数倍，
        num_workers：使用多进程加载的进程数，0代表不使用多进程
        shuffle:：是否将数据打乱
        dataloader是一个可迭代的对象，意味着我们可以像使用迭代器一样使用它 或者 or batch_datas, batch_labels in dataloader
        
        data.DataLoader():数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。在训练模型时使用到此函数，用来把训练数据分成多个小组，
        此函数每次抛出一组数据。直至把所有的数据都抛出。就是做一个数据的初始化
    '''
    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_batch = data.DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, drop_last=True)

    print("Training data shape", len(train_batch))
    print("Validation data shape", len(test_batch))

    # Model setting

    if args.ModelName == 'AE':
        model = AutoEncoderCov3D(args.c)
    elif args.ModelName == 'MemAE':
        model = AutoEncoderCov3DMem(args.c, args.MemDim, shrink_thres=args.ShrinkThres)
    elif args.ModelName == "ConvAEMem":
        model = ConvAEMem(args.c, args.t_length, args.MemDim, shrink_thres=args.ShrinkThres)
    else:
        model = []
        print('Wrong Name.')
    # print(model)
    model = model.cuda()  # 将模型加载到指定设备上
    parameter_list = [p for p in model.parameters() if p.requires_grad]

    # named_parameters():该方法可以输出模型的参数和该参数对应层的名字
    for name, p in model.named_parameters():
        if not p.requires_grad:
            print("---------NO GRADIENT-----", name)

    # 随机梯度下降优化算法
    optimizer = torch.optim.Adam(parameter_list, lr=args.lr)
    # 动态调整学习率 gamma-衰减率
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.2)  # version 2

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)

    # Report the training process
    log_dir = os.path.join(args.exp_dir, args.dataset_type, args.ModelName, 'lr_%.5f_entropyloss_%.5f_version_%d' % (
        args.lr, args.EntropyLossWeight, args.version))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    orig_stdout = sys.stdout
    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f

    print("The training path", train_dataset.video_name)
    print("The testing path", test_dataset.video_name)

    for arg in vars(args):
        print(arg, getattr(args, arg))

    # 将数据以特定的格式存储到对应的日志文件夹中
    train_writer = SummaryWriter(log_dir=log_dir)

    current_epoch = 0
    # 模型加载继续训练
    if args.continue_train:
        checkpoint = torch.load(args.resume)
        current_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f'Pre-trained generator and discriminator have been loaded.\n')
    else:
        # warmup
        # 保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数
        model.train()

        '''
            在pytorch中，tensor有一个requires_grad参数，如果设置为True，则反向传播时，该tensor就会自动求导。
            tensor的requires_grad的属性默认为False,若一个节点（叶子变量：自己创建的tensor）requires_grad被设置为True，
            那么所有依赖它的节点requires_grad都为True（即使其他相依赖的tensor的requires_grad = False）

            当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存
        '''
        # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
        with torch.no_grad():
            for batch_idx, frame in enumerate(train_batch):
                # frame = frame.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
                # frame = frame.permute(0, 2, 1, 3, 4)  # 维度换位
                frame = frame[:, 0:12]
                frame = frame.cuda()
                model_output = model(frame)  # model.forward(frame)

    # Training
    for epoch in range(current_epoch, args.epochs):
        model.train()
        tr_re_loss, tr_mem_loss, tr_tot = 0.0, 0.0, 0.0
        progress_bar = tqdm(train_batch)  # 训练进度条

        for batch_idx, frame in enumerate(progress_bar):
            progress_bar.update()
            # frame = frame.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
            # frame = frame.permute(0, 2, 1, 3, 4)
            target_frame = frame[:, 12:].cuda()
            input_frame = frame[:, 0:12].cuda()
            input_last = frame[:, 9:12].cuda()
            optimizer.zero_grad()   # 清空过往梯度

            model_output = model(input_frame)
            recons, attr = model_output['output'], model_output['att']

            re_loss = loss.get_reconstruction_loss(target_frame, recons, mean=0.5, std=0.5)  # 公式8
            mem_loss = loss.get_memory_loss(attr)  # 公式9
            tot_loss = re_loss + mem_loss * args.EntropyLossWeight  # 公式10
            tr_re_loss += re_loss.data.item()  # .data()-将variable变量变为tensor;  .item()-将tensor变为float
            tr_mem_loss += mem_loss.data.item()
            tr_tot += tot_loss.data.item()

            tot_loss.backward()  # 对函数进行反向传播，计算输出变量关于输入变量的梯度
            optimizer.step()   # 根据梯度更新网络参数

        train_writer.add_scalar("model/train-recons-loss", tr_re_loss / len(train_batch), epoch)
        train_writer.add_scalar("model/train-memory-sparse", tr_mem_loss / len(train_batch), epoch)
        train_writer.add_scalar("model/train-total-loss", tr_tot / len(train_batch), epoch)
        scheduler.step()   # # 根据梯度更新网络参数

        current_lr = optimizer.param_groups[0]['lr']
        train_writer.add_scalar('learning_rate', current_lr, epoch)
        torch.cuda.synchronize()
        if epoch % args.val_epoch == 0:
            model.eval()  # 不启用 BatchNormalization 和 Dropout
            re_loss_val, mem_loss_val = 0.0, 0.0
            for batch_idx, frame in enumerate(test_batch):
                # frame = frame.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
                # frame = frame.permute(0, 2, 1, 3, 4)
                # frame = frame.cuda()
                target_frame = frame[:, 12:]
                input_frame = frame[:, 0:12]
                target_frame = target_frame.cuda()
                input_frame = input_frame.cuda()
                model_output = model(input_frame)
                recons, attr = model_output['output'], model_output['att']
                re_loss = loss.get_reconstruction_loss(target_frame, recons, mean=0.5, std=0.5)
                mem_loss = loss.get_memory_loss(attr)
                re_loss_val += re_loss.data.item()
                mem_loss_val += mem_loss.data.item()
                if batch_idx == len(test_batch) - 1:
                    # detach():返回一个新的Tensor,但返回的结果是没有梯度的. cpu():把gpu上的数据转到cpu上. numpy():将tensor格式转为numpy
                    _input_npy = frame.detach().cpu().numpy()
                    _input_npy = _input_npy * 0.5 + 0.5
                    _recons_npy = recons.detach().cpu().numpy()
                    _recons_npy = _recons_npy * 0.5 + 0.5  # [batch_size, ch, time, imh, imw]
                    train_writer.add_images("image/input_image", arrange_image(_input_npy), epoch)
                    train_writer.add_images("image/reconstruction", arrange_image(_recons_npy), epoch)
            train_writer.add_scalar("model/val-recons-loss", re_loss_val / len(test_batch), epoch)
            train_writer.add_scalar("model/val-memory-sparse", mem_loss_val / len(test_batch), epoch)
            print("epoch %d" % epoch, "recons loss training %.4f validation %.4f" % (tr_re_loss, re_loss_val),
                  "memory sparsity training %.4f validation %.4f" % (tr_mem_loss, mem_loss_val))

        if epoch >= 40:
            state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "epoch": epoch}
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                torch.save(state_dict, log_dir + "/model-{:04d}.pt".format(epoch))

    sys.stdout = orig_stdout
    f.close()


if __name__ == '__main__':
    main()
