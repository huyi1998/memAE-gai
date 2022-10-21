import torch
from torch import nn


def get_memory_loss(memory_att):
    """The memory attribute should be with size [batch_size, memory_dim, reduced_time_dim, f_h, f_w]
    loss = \sum_{t=1}^{reduced_time_dim} (-mem) * (mem + 1e-12).log() 
    averaged on each pixel and each batch
    2. average over batch_size * fh * fw
    """
    s = memory_att.shape
    # 公式9
    memory_att = (-memory_att) * (memory_att + 1e-12).log()  # [batch_size, memory_dim, time, fh, fw]
    memory_att = memory_att.sum() / (s[0] * s[-2] * s[-1])  # average over batch_size * fh * fw
    return memory_att


# 归一化的逆过程
def get_unormalized_data(x_input, x_recons, mean, std):
    # input[channel] = (input[channel] - mean[channel]) / std[channel]
    x_input = x_input.mul(std).add(mean)
    x_recons = x_recons.mul(std).add(mean)
    return x_input, x_recons


def get_reconstruction_loss(x_input, x_recons, mean=0.5, std=0.5):
    """Calculates the reconstruction loss between x_input and x_recons
    x_input: [batch_size, ch, time, imh, imw]
    x_recons: [batch_size, ch, time, imh, imw]
    """
    # batch_size, ch, time_dimension, imh, imw = x_input.shape
    # x_input, x_recons = get_unormalized_data(x_input, x_recons, mean, std)
    # recons_loss = (x_input - x_recons) ** 2
    # recons_loss = recons_loss.sum().sqrt() / (batch_size * imh * imw)
    # return recons_loss
    batch_size, ch, imh, imw = x_input.shape
    x_input, x_recons = get_unormalized_data(x_input, x_recons, mean, std)
    recons_loss = (x_input - x_recons) ** 2
    recons_loss = recons_loss.sum().sqrt() / (batch_size * imh * imw)
    return recons_loss


# 光流损失
def flow_loss(gen_flows, gt_flows):
    return torch.mean(torch.abs(gen_flows - gt_flows))


def triplet_loss(anchor, positive, negative):
    triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    return triplet(anchor, positive, negative)
