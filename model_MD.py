import torch
from torch import nn
from modules_MD import ConvSC
from cls_MD.mmcls.models.backbones import MDATranslator

import torch.nn.functional as F


def stride_generator(N, reverse=False):  
    strides = [1, 2, 1, 2] * 10  
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]

class Encoder(nn.Module):  
    def __init__(self, C_in, C_hid, N_S, kernel_size):  ## para: [1,16,4]
        super(Encoder, self).__init__()
        global grad_CAM  # 再到里面呼叫外面的参数
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0], kernel_size=kernel_size[0], padding=(kernel_size[0]-1)//2),  # 两种步长，1和2.但是1和2都会使ConvSC的transpose为false
            *[ConvSC(C_hid, C_hid, stride=strides[i], kernel_size=kernel_size[i], padding=(kernel_size[i]-1)//2) for i in range(1,N_S)]
        )

        self.norm = nn.Sequential(nn.BatchNorm2d(C_hid),
                                  nn.BatchNorm2d(C_hid),
                                  nn.BatchNorm2d(C_hid),
                                  nn.BatchNorm2d(C_hid))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  

        out = []
        out.append(self.enc[0](x))
        for i in range(1, len(self.enc)):
            out.append(self.enc[i](out[i-1])) 
        return out


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, T_in, T_out, N_S, kernel_size):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S) 
        self.T_in = T_in
        self.C_hid = C_hid
        self.T_out = T_out
        self.C_out = C_out

        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=strides[i], kernel_size=kernel_size[i], padding=(kernel_size[i] - 1) // 2,
                     transpose=True) for i in range(N_S - 1, 0, -1)],
            ConvSC(C_hid, C_hid, stride=strides[0], kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2,
                   transpose=True)
        )

        self.conv_cat = nn.Sequential(nn.Conv2d(2 * C_hid, C_hid, kernel_size=1, stride=1, padding=0),
                                      nn.Conv2d(2 * C_hid, C_hid, kernel_size=1, stride=1, padding=0),
                                      nn.Conv2d(2 * C_hid, C_hid, kernel_size=1, stride=1, padding=0))

        self.norm = nn.Sequential(nn.BatchNorm2d(C_hid),
                                  nn.BatchNorm2d(C_hid),
                                  nn.BatchNorm2d(C_hid),
                                  nn.BatchNorm2d(C_hid))
        self.relu = nn.ReLU(inplace=True)
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        self.channel_ex = nn.Conv2d(self.T_in * self.C_hid, self.T_out * self.C_out, 1)

    def forward(self, x):
        data = x[-1]

        for i in range(0, len(self.dec) - 1):
            out = self.dec[i](data)

            data = out + x[-(i + 2)] 

        Y = self.dec[-1](data)
        BT, _, H, W = Y.shape

        Y = self.readout(Y)
        return Y


class MDAT(nn.Module):
    def __init__(self, channel_in, channel_hid, N_S, incep_ker=[3, 5, 7, 11], groups=8, rednet_deep=26,
                 layer_config=(2, 8, 2, 8)):  
        super(MDAT, self).__init__()
        print(groups)
        self.net = MDATranslator(layers=N_S, layer_config=layer_config, in_channels=channel_in, reduction=groups, data_size=64, g1=2, g2=4).to("cuda")
        self.channel_in = channel_in  


    def forward(self, x):

        for i in range(len(x)):
            BT, _, H, W = x[i].shape 
            x[i] = x[i].reshape(-1, self.channel_in, H, W) 
        x = self.net(x)

        for i in range(len(x)):
            _, _, H, W = x[i].shape
            x[i] = x[i].reshape(BT, -1, H, W)

        return x


class MDANet(nn.Module):
    def __init__(self, shape_in, shape_out, hid_S=64, hid_T=256, N_S=4, kernel_size=[3, 5, 7, 5], N_T=8, rednet_deep=26,
                 layer_config=(1, 8, 2, 8), incep_ker=[3, 5, 7, 11], groups=8):
        super(MDANet, self).__init__()
        T, C, H, W = shape_in
        self.T_out, self.C_out, _, _ = shape_out


        self.enc = Encoder(C, hid_S, N_S, kernel_size)
        self.MDTranslator = MDAT(T * hid_S, hid_T, N_S, incep_ker, groups, rednet_deep, layer_config)
        self.dec = Decoder(hid_S, self.C_out, T, self.T_out, N_S, kernel_size)


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape

        x = x_raw.contiguous().view(B * T, C, H, W)
        data = self.enc(x) 
        data = self.MDTranslator(data) 

        Y = self.dec(data)

        Y = Y.reshape(B, self.T_out, self.C_out, H, W)
        return Y
