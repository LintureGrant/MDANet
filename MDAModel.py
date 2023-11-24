from torch import nn
from MDAModules import ConvSC
from cls_MD.mmcls.models.backbones import TModule


def stride_generator(N, reverse=False):  
    strides = [1, 2, 1, 2] * 10  
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]

class Encoder(nn.Module):  
    def __init__(self, C_in, C_hid, layer_num, kernel_size):  ## para: [1,16,4]
        super(Encoder, self).__init__()
        global grad_CAM  # 再到里面呼叫外面的参数
        strides = stride_generator(layer_num)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0], kernel_size=kernel_size[0], padding=(kernel_size[0]-1)//2),  # 两种步长，1和2.但是1和2都会使ConvSC的transpose为false
            *[ConvSC(C_hid, C_hid, stride=strides[i], kernel_size=kernel_size[i], padding=(kernel_size[i]-1)//2) for i in range(1,layer_num)]
        )

    def forward(self, x):  

        out = []
        out.append(self.enc[0](x))
        for i in range(1, len(self.enc)):
            out.append(self.enc[i](out[i-1])) 
        return out


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, T_in, T_out, layer_num, kernel_size):
        super(Decoder, self).__init__()
        strides = stride_generator(layer_num)
        self.T_in = T_in
        self.C_hid = C_hid
        self.T_out = T_out
        self.C_out = C_out

        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=strides[i], kernel_size=kernel_size[i], padding=(kernel_size[i] - 1) // 2,
                     transpose=True) for i in range(layer_num - 1, 0, -1)],
            ConvSC(C_hid, C_hid, stride=strides[0], kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2,
                   transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)


    def forward(self, x):
        data = x[-1]

        for i in range(0, len(self.dec) - 1):
            out = self.dec[i](data)

            data = out + x[-(i + 2)] 

        Y = self.dec[-1](data)
        BT, _, H, W = Y.shape

        Y = self.readout(Y)
        return Y


class MDATranslator(nn.Module):
    def __init__(self, in_channel, layer_num, data_size, g1, g2, reduction=8, layer_config=(2, 8, 2, 8)):
        super(MDATranslator, self).__init__()
        self.net = TModule(layer=layer_num,
                           layer_config=layer_config,
                           in_channels=in_channel,
                           reduction=reduction,
                           data_size=data_size,
                           g1=g1,
                           g2=g2,
                           backbone='MDAUnit').to("cuda")
        self.in_channel = in_channel

    def forward(self, x):

        for i in range(len(x)):
            BT, _, H, W = x[i].shape 
            x[i] = x[i].reshape(-1, self.in_channel, H, W)
        x = self.net(x)

        for i in range(len(x)):
            _, _, H, W = x[i].shape
            x[i] = x[i].reshape(BT, -1, H, W)

        return x


class MDANet(nn.Module):
    def __init__(self,
                 shape_in,
                 shape_out,
                 hid_channel=64,
                 layer_num=4,
                 kernel_size=(3, 3, 3, 3),
                 layer_config=(1, 8, 2, 8),
                 reduction=8,
                 group_param=(2, 4)):
        super(MDANet, self).__init__()

        T, C, H, W = shape_in

        self.T_out, self.C_out, _, _ = shape_out

        self.enc = Encoder(C, hid_channel, layer_num, kernel_size)

        self.MDTranslator = MDATranslator(in_channel=T * hid_channel,
                                          layer_num=layer_num,
                                          reduction=reduction,
                                          data_size=H,
                                          g1=group_param[0],
                                          g2=group_param[1],
                                          layer_config=layer_config)

        self.dec = Decoder(hid_channel, self.C_out, T, self.T_out, layer_num, kernel_size)


    def forward(self, x_raw):

        B, T, C, H, W = x_raw.shape

        x = x_raw.contiguous().view(B * T, C, H, W)

        data = self.enc(x)

        data = self.MDTranslator(data)

        Y = self.dec(data)

        Y = Y.reshape(B, self.T_out, self.C_out, H, W)

        return Y
