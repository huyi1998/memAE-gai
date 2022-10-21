from __future__ import absolute_import, print_function
import torch
from torch import nn
from models import MemModule


class MemoryAutoEncoder(nn.Module):
    def __init__(self, chnum_in):
        super(MemoryAutoEncoder, self).__init__()

        def Basic(intInput, intOutput):
            return nn.Sequential(nn.Conv3d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3, 3),
                                           stride=(1, 2, 2), padding=(1, 1, 1)),
                                 nn.BatchNorm3d(intOutput),
                                 nn.LeakyReLU(0.2, inplace=True))

        def Basic_(intInput, intOutput):
            return nn.Sequential(nn.Conv3d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3, 3),
                                           stride=(2, 2, 2), padding=(1, 1, 1)),
                                 nn.BatchNorm3d(intOutput),
                                 nn.LeakyReLU(0.2, inplace=True))

        self.moduleConv1 = Basic(chnum_in, 96)
        self.moduleConv2 = Basic_(96, 128)
        self.moduleConv3 = Basic_(128, 256)
        self.moduleConv4 = Basic_(256, 256)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)            # [1,1,8,256,256] -> [1,96,8,128,128]
        tensorConv2 = self.moduleConv2(tensorConv1)  # [1,96,8,128,128] -> [1,128,4,64,64]
        tensorConv3 = self.moduleConv3(tensorConv2)  # [1,128,4,64,64] -> [1,256,2,32,32]
        tensorConv4 = self.moduleConv4(tensorConv3)  # [1,256,2,32,32] -> [1,236,1,16,16]
        return tensorConv4


class MemoryAutoDecoder(nn.Module):
    def __init__(self):
        super(MemoryAutoDecoder, self).__init__()

        def Gen(intInput, intOutput):
            return nn.Sequential(nn.ConvTranspose3d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3, 3),
                                                    stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
                                 nn.BatchNorm3d(intOutput),
                                 nn.LeakyReLU(0.2, inplace=True))

        def Gen_(intInput, intOutput):
            return nn.ConvTranspose3d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3, 3),
                                      stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1))

        self.moduleConv1 = Gen(256, 256)
        self.moduleConv2 = Gen(256, 128)
        self.moduleConv3 = Gen(128, 96)
        self.moduleConv4 = Gen_(96, 1)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)            # [1,256,1,16,16] -> [1,256,2,32,32]
        tensorConv2 = self.moduleConv2(tensorConv1)  # [1,256,2,32,32] -> [1,128,4,64,64]
        tensorConv3 = self.moduleConv3(tensorConv2)  # [1,128,4,64,64] -> [1,96,8,128,128]
        tensorConv4 = self.moduleConv4(tensorConv3)  # [1,96,8,128,128] -> [1,1,8,256,256]
        return tensorConv4


class ConvAE(nn.Module):
    def __init__(self, chnum_in, mem_dim, shrink_thres=0.0025):
        super(ConvAE, self).__init__()
        self.encoder = MemoryAutoEncoder(chnum_in)
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=256, shrink_thres=shrink_thres)
        self.decoder = MemoryAutoDecoder()

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'output': output, 'att': att}


if __name__ == "__main__":
    model = ConvAE(1, 2000, 0.0025)
    input = torch.ones(1, 1, 4, 256, 256)  # [batch_size,c,t_length,h,w]
    output = model(input)
