# from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, bias=True):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    # def __init__(self, in_ch, out_ch):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, bias=True):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Input _ [batch * channel(# of channels of each image) * depth(# of frames) * height * width].
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64 #TODO: original paper starts with 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 64,128,256,512,1024

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        # print("unet")
        # print(x.shape)
        # print(padded.shape)

        e1 = self.Conv1(x)
        # print("conv1:")
        # print(e1.shape)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        # print("conv2:")
        # print(e2.shape)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        # print("conv3:")
        # print(e3.shape)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        #print("conv4:")
        #print(e4.size())

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        #print("conv5:")
        #print(e5.size())

        d5 = self.Up5(e5)
        #print("d5:")
        #print(d5.size())
        # print("e4:")
        # print(e4.shape)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        # print("upconv5:")
        # print(d5.size)

        d4 = self.Up4(d5)
        # print("d4:")
        # print(d4.shape)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        # print("upconv4:")
        # print(d4.shape)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        # print("upconv3:")
        # print(d3.shape)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # print("upconv2:")
        # print(d2.shape)
        out = self.Conv(d2)
        # print("out:")
        # print(out.shape)
        # d1 = self.active(out)
        # indx = torch.gt(out, 0.5).float()
        # indx1 = torch.le(out, 0.5).float()
        # out = torch.tensor([[torch.sum(model_out), torch.sum(1.0 - model_out)]]).to(self.device)
        # out =  torch.unique(model_out, return_counts = True)[1].unsqueeze(0).float()
        # op1 = ((out / out) * indx).sum(dim=(2, 3, 4))
        # op2 = ((out / out) * indx1).sum(dim=(2, 3, 4))
        # return torch.cat((op2, op1), dim=1)
        return [out, d3,d4]

class U_Net_DeepSup(nn.Module):
    """
    UNet - Basic Implementation
    Input _ [batch * channel(# of channels of each image) * depth(# of frames) * height * width].
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=1):
        super(U_Net_DeepSup, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 64,128,256,512,1024

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        #1x1x1 Convolution for Deep Supervision
        self.Conv_d3 = conv_block(filters[1], 1)
        self.Conv_d4 = conv_block(filters[2], 1)



        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        # print("unet")
        # print(x.shape)
        # print(padded.shape)


        # if isinstance(x, tuple):
        #     x = list(x)
        #     x = torch.stack(x).squeeze(0)
        # # print(x.shape)
        # # print(type(x))
        e1 = self.Conv1(x)
        # print("conv1:")
        # print(e1.shape)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        # print("conv2:")
        # print(e2.shape)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        # print("conv3:")
        # print(e3.shape)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        # print("conv4:")
        # print(e4.shape)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        # print("conv5:")
        # print(e5.shape)

        d5 = self.Up5(e5)
        # print("d5:")
        # print(d5.shape)
        # print("e4:")
        # print(e4.shape)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        # print("upconv5:")
        # print(d5.size)

        d4 = self.Up4(d5)
        # print("d4:")
        # print(d4.shape)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4_out  = self.Conv_d4(d4)
        
                
        # print("upconv4:")
        # print(d4.shape)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)        
        d3_out  = self.Conv_d3(d3)

        # print("upconv3:")
        # print(d3.shape)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # print("upconv2:")
        # print(d2.shape)
        out = self.Conv(d2)
        # print("out:")
        # print(out.shape)
        # d1 = self.active(out)

        return [out, d3_out , d4_out]


