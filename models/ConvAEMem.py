from __future__ import absolute_import, print_function
import torch
from torch import nn
from models import MemModule


class ConvAEEncoder(nn.Module):
    def __init__(self, n_channel=3, t_length=5):
        super(ConvAEEncoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)  # [3,12,256,256] -> [3,64,256,256]
        tensorPool1 = self.modulePool1(tensorConv1)  # [3,64,256,256] -> [3,64,128,128]

        tensorConv2 = self.moduleConv2(tensorPool1)  # [3,64,128,128] -> [3,128,128,128]
        tensorPool2 = self.modulePool2(tensorConv2)  # [3,128,128,128] -> [3,128,64,64]

        tensorConv3 = self.moduleConv3(tensorPool2)  # [3,128,64,64] -> [3,256,64,64]
        tensorPool3 = self.modulePool3(tensorConv3)  # [3,256,64,64] -> [3,256,32,32]

        tensorConv4 = self.moduleConv4(tensorPool3)  # [3,256,32,32] -> [3,512,32,32]

        return tensorConv4, tensorConv1, tensorConv2, tensorConv3


class ConvAEDecoder(nn.Module):
    def __init__(self, n_channel=3, t_length=5):
        super(ConvAEDecoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128, n_channel, 64)

    def forward(self, x, skip1, skip2, skip3):
        tensorConv = self.moduleConv(x)  # [3,1024,32,32] -> [3,512,32,32]

        tensorUpsample4 = self.moduleUpsample4(tensorConv)  # [3,512,32,32] -> [3,256,64,64]
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)  # [3,256,64,64]+[3,256,64,64] -> [3,512,64,64]

        tensorDeconv3 = self.moduleDeconv3(cat4)  # [3,512,64,64] -> [3,256,64,64]
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)  # [3,256,64,64] -> [3,128,128,128]
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)  # [3,128,128,128]+[3,128,128,128] -> [3,256,128,128]

        tensorDeconv2 = self.moduleDeconv2(cat3)  # [3,256,128,128] -> [3,128,128,128]
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)  # [3,128,128,128] -> [3,64,256,256]
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)  # [3,64,256,256]+[3,64,256,256] -> [3,128,256,256]

        output = self.moduleDeconv1(cat2)  # [3,128,256,256] -> [3,3,256,256]

        return output


class ConvAEMem(nn.Module):
    def __init__(self, chnum_in, t_length, mem_dim, shrink_thres=0.0025):
        super(ConvAEMem, self).__init__()
        self.encoder = ConvAEEncoder(chnum_in, t_length)
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=512, shrink_thres=shrink_thres)
        self.decoder = ConvAEDecoder(chnum_in, t_length)

    # def forward(self, x):
    #     f, skip1, skip2, skip3 = self.encoder(x)
    #     res_mem = self.mem_rep(f)
    #     f = res_mem['output']
    #     att = res_mem['att']
    #     output = self.decoder(f, skip1, skip2, skip3)
    #     return {'output': output, 'att': att}
    #
    def forward(self, x):
        f, skip1, skip2, skip3 = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output_good']
        f2 = res_mem['output_bad']
        att = res_mem['att']
        output_good = self.decoder(f, skip1, skip2, skip3)
        output_bad = self.decoder(f2, skip1, skip2, skip3)
        return {'output_good': output_good, 'output_bad':output_bad, 'att': att}


if __name__ == "__main__":
    model = ConvAEMem(3, 5, 2000, 0.0025)
    input = torch.ones(4, 12, 256, 256)  # [batch_size,c,t_length,h,w]
    output = model(input)  # [4,3,256,256]、[4,2000,32,32]
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    # output = triplet_loss(anchor, positive, negative)
    # output.backward()