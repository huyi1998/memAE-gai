from __future__ import absolute_import, print_function
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        #         print("memory shape", self.weight.shape)
        self.bias = None
        self.shrink_thres = shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # def forward(self, input):
    #     att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (NxC) x (CxM) = NxM    公式5
    #     att_weight = F.softmax(att_weight, dim=1)  # NxM   公式4（Wi）
    #     # ReLU based shrinkage, hard shrinkage for positive value
    #     if self.shrink_thres > 0:
    #         att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)  # 公式6、7
    #         # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
    #         # normalize???
    #         att_weight = F.normalize(att_weight, p=1, dim=1)  # Lp归一化
    #         # att_weight = F.softmax(att_weight, dim=1)
    #         # att_weight = self.hard_sparse_shrink_opt(att_weight)
    #     mem_trans = self.weight.permute(1, 0)  # Mem^T, (MxC)^T
    #     output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (NxM) x (MxC) = NxC   公式3
    #     return {'output': output, 'att': att_weight}  # output, att_weight

    def forward(self, input):
        att_weight_bad = F.linear(input, self.weight)  # Fea x Mem^T, (NxC) x (CxM) = NxM    公式5
        att_weight_bad = F.softmax(att_weight_bad, dim=1)  # NxM   公式4（Wi）
        att_weight_good = att_weight_bad.contiguous()
        # ReLU based shrinkage, hard shrinkage for positive value
        if self.shrink_thres > 0:
            att_weight_good = hard_shrink_relu(att_weight_good, lambd=self.shrink_thres)  # 公式6、7
            att_weight_good = F.normalize(att_weight_good, p=1, dim=1)  # Lp归一化
        mem_trans = self.weight.permute(1, 0)  # Mem^T, (MxC)^T
        output_good = F.linear(att_weight_good, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (NxM) x (MxC) = NxC   公式3
        output_bad = F.linear(att_weight_bad, mem_trans)
        return {'output_good': output_good, 'output_bad': output_bad, 'att': att_weight_good}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)  # [batch_size, ch, time_length, imh, imw]

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)     # [batch_size, imh, imw, ch]
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)  # [batch_size, time, imh, imw, ch]
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()  # 深拷贝
        # view()和reshape()都可以用来改变Tensor的形状，区别是view只能对满足连续条件的Tensor进行操作，而reshape可以对不满足连续条件的Tensor进行操作
        # -1表示根据另外一个数来自动调整维度
        x = x.view(-1, s[1])  # [batch_size * time * imh * imw, ch]  [N x C]

        # y_and = self.memory(x)
        # y = y_and['output']
        # att = y_and['att']
        #
        # if l == 3:
        #     y = y.view(s[0], s[2], s[1])
        #     y = y.permute(0, 2, 1)
        #     att = att.view(s[0], s[2], self.mem_dim)
        #     att = att.permute(0, 2, 1)
        # elif l == 4:
        #     y = y.view(s[0], s[2], s[3], s[1])
        #     y = y.permute(0, 3, 1, 2)
        #     x = x.view(s[0], s[2], s[3], s[1])
        #     x = x.permute(0, 3, 1, 2)
        #     y = torch.cat((y, x), dim=1)
        #     att = att.view(s[0], s[2], s[3], self.mem_dim)
        #     att = att.permute(0, 3, 1, 2)
        # elif l == 5:
        #     y = y.view(s[0], s[2], s[3], s[4], s[1])
        #     y = y.permute(0, 4, 1, 2, 3)
        #     att = att.view(s[0], s[2], s[3], s[4],
        #                    self.mem_dim)  # [batch_size, time_length, imh, imw, memory_dimension]
        #     att = att.permute(0, 4, 1, 2, 3)  # [batch_size, memory_dimension, time_length, imh, imw]
        # else:
        #     y = x
        #     att = att
        #     print('wrong feature map size')
        # return {'output': y, 'att': att}

        y_and = self.memory(x)
        y_good = y_and['output_good']
        y_bad = y_and['output_bad']
        att = y_and['att']

        if l == 4:
            y_good = y_good.view(s[0], s[2], s[3], s[1])
            y_good = y_good.permute(0, 3, 1, 2)
            y_bad = y_bad.view(s[0], s[2], s[3], s[1])
            y_bad = y_bad.permute(0, 3, 1, 2)
            x = x.view(s[0], s[2], s[3], s[1])
            x = x.permute(0, 3, 1, 2)
            y_good = torch.cat((y_good, x), dim=1)
            y_bad = torch.cat((y_bad, x), dim=1)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return {'output_good': y_good, 'output_bad': y_bad, 'att': att}


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    # 公式7  relu=max(x,0)
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output
