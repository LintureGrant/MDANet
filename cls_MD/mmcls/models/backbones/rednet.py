import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from ..utils.involution_cuda import involution, My_involution
from ..utils.involution_cuda import _involution_cuda
import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_
import math
#from self_attention_cv.bottleneck_transformer import BottleneckBlock

def softmax(x):
    e_x = np.exp((x-torch.max(x)).detach().numpy())# 防溢出
    return e_x/e_x.sum(0)

# class ScaledDotProductAttention(nn.Module):
#     def __init__(self,  channel, fig_size,count):
#         super().__init__()
#         self.scale = 1
#         self.channel = channel
#         #定义三个可学习的参数矩阵
#         self.Wq = torch.nn.Parameter(torch.randn((fig_size, fig_size), requires_grad=True))
#         self.Wk = torch.nn.Parameter(torch.randn((fig_size, fig_size), requires_grad=True))
#         self.Wv = torch.nn.Parameter(torch.randn((fig_size, fig_size), requires_grad=True))
#         self.register_parameter("Wq"+str(count), self.Wq)
#         self.register_parameter("Wk"+str(count), self.Wk)
#         self.register_parameter("Wi"+str(count), self.Wv)
#
#         # self.avapool = nn.AvgPool2d(kernel_size=fig_size, stride= fig_size, padding=0)
#         # self.trans_matrix = torch.zeros([fig_size * channel,channel],dtype=torch.float).to("cuda")
#         # for i in range(channel):
#         #     for j in range(fig_size):
#         #         self.trans_matrix[i * fig_size + j, i] = 1
#     def forward(self, data, mask=None):
#         N, C, H, W = data.shape
#         #计算q,k,v
#         self.Wq = self.Wq.to('cuda')
#         self.Wk = self.Wk.to('cuda')
#         self.Wv = self.Wv.to('cuda')
#
#         q = torch.matmul(self.Wq, data).to('cuda')
#         k = torch.matmul(self.Wk, data).to('cuda')
#         v = torch.matmul(self.Wv, data).to('cuda')
#         # 计算q，k的内积，输出应为[10,1,1]
#         k = k.reshape(N*C*H, W).to('cuda')
#         q = q.reshape(H, N*W*C).to('cuda')
#         A = torch.mm(k, q).to('cuda')  # 640*640的矩阵
#         # A = torch.squeeze(self.avapool(A.cpu().unsqueeze(0).unsqueeze(1)))#扩充两次维度以便使用池化
#
#         A /= np.sqrt(self.channel*self.channel) #
#         A_hat = torch.tensor(softmax(A.cpu())).to('cuda')
#
#         #######  下面是需要修改的地方   ########
#         # output = v.reshape(H, N*W*C) @ (self.trans_matrix.to('cuda') @ A_hat.to('cuda') @ self.trans_matrix.T.to('cuda'))
#         output = v.reshape(H, N*W*C) @ A_hat.to('cuda')
#         # u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul  公式q*k转置，这里出来的u就是一个数字了
#         # u = u / self.scale # 2.Scale
#         #
#         # attn = self.softmax(u) # 4.Softmax
#         # output = torch.bmm(attn, v) # 5.Output
#         output = output.reshape(N, C, H, W).to('cuda')
#         return output



class MEModule(nn.Module):
    """ Motion exciation module

    :param reduction=16
    :param n_segment=8/16
    """

    def __init__(self, channel, reduction=1, n_segment=10):
        super(MEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        # self.conv1 = involution(self.channel,kernel_size=7,stride=1)
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel // self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        # self.conv2 = involution(self.channel,kernel_size=15,stride=1)
        self.conv2 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel // self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel // self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
        # self.conv3 = involution(self.channel,kernel_size=7,stride=1)
        self.conv3 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.identity = nn.Identity()

    def forward(self, x):
        B, Tc, H, W = x.size()
        x = x.view(-1, self.channel, H, W)
        nt, c, h, w = x.size()
        bottleneck = self.conv1(x)  # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck)  # nt, c//r, h, w

        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
        t_fea, __ = reshape_bottleneck.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w

        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea + t_fea  # n, t-1, c//r, h, w
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
        diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  # nt, c//r, h, w
        y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
        y = self.conv3(y)  # nt, c, 1, 1
        y = self.bn3(y)  # nt, c, 1, 1
        y = self.sigmoid(y)  # nt, c, 1, 1
        y = y - 0.5
        output = x + x * y.expand_as(x)
        return output

# #
class PConv(nn.Module):
    def __init__(self,
                 dim: int,
                 out_dim: int,
                 n_div: int,
                 forward: str="split_cat",
                 kernel_size: int=3)->None:
        super().__init__()
        self.dim_conv = dim //n_div
        self.dim_untouched = dim - self.dim_conv

        self.conv = nn.Conv2d(self.dim_conv,
                              self.dim_conv,
                              kernel_size,
                              stride=1,
                              padding=(kernel_size-1)//2,
                              bias=False)
        self.convone = nn.Conv2d(self.dim_untouched ,
                                 out_dim - self.dim_conv,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        if forward =="slicing":
            self.forward = self.self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
    def forward_slicing(self, x):
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x
    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x2 = self.convone(x2)
        x = torch.cat((x1,x2),1)
        return x


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class WeightedPermuteMLP(nn.Module):
#     def __init__(self, dim, c_dim, len_sequence, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#                  proj_drop=0.):
#         super().__init__()
#         self.segment_dim = segment_dim
#         self.T = len_sequence
#         self.ratio = (self.T - 1) / 2
#         self.mlp_c = nn.Linear(int(c_dim * (self.T - 1) / 2), c_dim, bias=qkv_bias)
#         self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
#         self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
#
#         self.reweight = Mlp(c_dim, c_dim // 4, c_dim * 3)
#
#         self.proj = nn.Linear(c_dim, c_dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.norm1 = nn.LayerNorm(c_dim)
#         self.norm2 = nn.LayerNorm(c_dim)
#         self.mlp = Mlp(in_features=c_dim, hidden_features=c_dim // 4, act_layer=nn.GELU)
#         #         self.conv = nn.Conv2d(in_channels=c_dim,
#         #                               out_channels=c_dim,
#         #                               kernel_size=1,
#         #                               stride=1,
#         #                               padding=(1-1)//2)
#         #         self.inv = involution(c_dim, 15, 1)
#         #         self.act = nn.ReLU(inplace=True)
#         self.norm3 = nn.LayerNorm(c_dim)
#
#     def multi_diff(self, data):
#         B, H, W, C = data.shape
#         data = data.reshape(B * C // self.T, H, W, self.T)
#         hid = torch.zeros(B * C // self.T, H, W, 1).to('cuda')
#         # data = torch.cat([data[:,0].unsqueeze(1), data], dim=1)
#         for i in range(0, self.T):
#             hid = torch.cat([hid, data[:, :, :, 0: self.T - i - 1] - data[:, :, :, i + 1: 10]], dim=3)
#
#         return hid[:, :, :, 1:].reshape(B, H, W, -1)
#
#     def forward(self, x):
#         x = x.permute(0, 3, 2, 1)
#
#         B, H, W, C = x.shape
#
#         x = self.norm1(x)
#         diff = self.multi_diff(x)
#
#         S = C // self.segment_dim
#         h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H * S)
#         h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
#
#         w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W * S)
#         w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)
#
#         c = self.mlp_c(diff)
#
#         a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
#         a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
#
#         x = h * a[0] + w * a[1] + c * a[2]
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         # res = x
#         #
#         # x = self.mlp(self.norm2(x))
#         # x = x + res
#         x = x.permute(0, 3, 2, 1)
#         return x


##############################################              right  mse  15.  neer 16  ############################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop(x)
        return x



class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim, c_dim, len_sequence, segment_dim=8, input_size=64, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim
        self.T = len_sequence
        self.ratio = (self.T - 1) / 2
        self.mlp_c = nn.Linear(c_dim, c_dim, bias=qkv_bias)
        self.conv_c = nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1,stride=1)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

        self.v1_h = nn.Linear(c_dim, c_dim, bias=qkv_bias)
        self.v1_w = nn.Linear(c_dim, c_dim, bias=qkv_bias)

        self.reweight_h = Mlp(c_dim, c_dim // 4, c_dim)
        self.reweight_w = Mlp(c_dim, c_dim // 4, c_dim)
        self.reweight_c1 = Mlp(input_size, input_size // 4, input_size)
        self.reweight_c2 = Mlp(input_size, input_size // 4, input_size)

        self.proj = nn.Linear(c_dim, c_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm1 = nn.LayerNorm(c_dim)
        self.norm2 = nn.LayerNorm(c_dim)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = Mlp(in_features=int(c_dim * self.ratio), hidden_features=c_dim // 4, out_features=c_dim,act_layer=nn.ReLU)
        self.drop_path = DropPath(0.1)


    def multi_diff(self, data):
        B, H, W, C = data.shape
        data = data.reshape(B * C // self.T, H, W, self.T)
        hid = torch.zeros(B * C // self.T, H, W, 1).to('cuda')
        # data = torch.cat([data[:,0].unsqueeze(1), data], dim=1)
        for i in range(0, 10):
            hid = torch.cat([hid, data[:, :, :, 0: self.T - i - 1] - data[:, :, :, i + 1:]], dim=3)

        return hid[:, :, :, 1:].reshape(B, H, W, -1)

    def forward(self, x):

        x = x.permute(0, 3, 2, 1)

        B, H, W, C = x.shape

        x = self.norm1(x)

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim * W, H * S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        # w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H * self.segment_dim, W * S)
        # w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)

        a = (h).mean(1)
        # b = (w).mean(2)
        ce = (c).mean(3)
        a = self.reweight_h(a).reshape(B, 1, W, C)
        # b = self.reweight_w(b).reshape(B, H, 1, C)
        # ce1 = self.reweight_c1(ce.reshape(B, H, W)).reshape(B, H, W, 1)
        # ce2 = self.reweight_c2(ce.reshape(B, H, W).permute(0,2,1)).permute(0,2,1).reshape(B, H, W, 1)
        # a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).unsqueeze(2).unsqueeze(2)

        # x = h * (b.expand_as(h)) + w * ( a.expand_as(w)) + (c * (a + b)) # + ce.expand_as(c) * (a.expand_as(c) + b.expand_as(c))
        x = h * (a.expand_as(h)) + (c * (a))
        x = (x).permute(0, 3, 2, 1)

        return x


# class WeightedPermuteMLP(nn.Module):
#     def __init__(self, dim, c_dim, len_sequence, segment_dim=8, input_size=64, qkv_bias=False, qk_scale=None, attn_drop=0.,
#                  proj_drop=0.):
#         super().__init__()
#         self.segment_dim = segment_dim
#         self.T = len_sequence
#         self.ratio = (self.T - 1) / 2
#         self.mlp_c = nn.Linear(int(c_dim), c_dim, bias=qkv_bias)
#         self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
#         self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
#
#         self.mlp_h_norm = nn.LayerNorm(dim)
#         self.mlp_w_norm = nn.LayerNorm(dim)
#         self.mlp_c_norm = nn.LayerNorm(c_dim)
#
#         self.mlp_h_act = nn.ReLU(inplace=True)
#         self.mlp_w_act = nn.ReLU(inplace=True)
#         self.mlp_c_act = nn.ReLU(inplace=True)
#
#         self.v1_h = nn.Linear(c_dim, c_dim, bias=qkv_bias)
#         self.v1_w = nn.Linear(c_dim, c_dim, bias=qkv_bias)
#
#         self.reweight_h = Mlp(dim, dim // 4, dim )
#         self.reweight_w = Mlp(dim, dim // 4, dim )
#         self.reweight = Mlp(c_dim, c_dim // 4, c_dim * 3)
#
#         self.proj = nn.Linear(c_dim, c_dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.norm1 = nn.LayerNorm(c_dim)
#         self.norm2 = nn.LayerNorm(c_dim)
#         self.mlp = Mlp(in_features=int(c_dim * self.ratio), hidden_features=c_dim // 4, out_features=c_dim,act_layer=nn.ReLU)
#         self.drop_path = DropPath(0.1)
#
#         self.unflod = nn.Unfold((3, input_size * c_dim//self.segment_dim), stride=(2,1))
#         # nn.MultiheadAttention
#         # self.attention = SelfAttention(1, input_size * self.segment_dim, input_size * c_dim//self.segment_dim, 0.1)
#         self.wq = nn.Linear(dim, dim)
#         self.wk = nn.Linear(dim, dim)
#         self.wv = nn.Linear(dim, dim)
#         self.atten_dropout = nn.Dropout(0.)
#
#     def multi_diff(self, data):
#         B, H, W, C = data.shape
#         data = data.reshape(B * C // self.T, H, W, self.T)
#         hid = torch.zeros(B * C // self.T, H, W, 1).to('cuda')
#         # data = torch.cat([data[:,0].unsqueeze(1), data], dim=1)
#         for i in range(0, 10):
#             hid = torch.cat([hid, data[:, :, :, 0: self.T - i - 1] - data[:, :, :, i + 1:]], dim=3)
#
#         return hid[:, :, :, 1:].reshape(B, H, W, -1)
#
#     def forward(self, x):
#         res = x
#
#         x = x.permute(0, 3, 2, 1)
#
#         B, H, W, C = x.shape
#
#         x = self.mlp_c(x)
#         x = self.mlp_c_norm(x)
#         x = self.mlp_c_act(x)
#
#         S = C // self.segment_dim
#         h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H * S)
#         h = self.reweight_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
#
#         w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W * S)
#         w = self.reweight_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)
#
#
#         #
#         # a = (h).mean(1)
#         # b = (w).mean(2)
#         # a = self.reweight_h(a ).reshape(B, H, C, 1).permute(3, 0, 1, 2).unsqueeze(2)
#         # b = self.reweight_w(b ).reshape(B, W, C, 1).permute(3, 0, 1, 2).unsqueeze(3)
#         # # a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).unsqueeze(2).unsqueeze(2)
#         #
#         # x = h * (a[0].expand_as(h) ) + w * ( b[0].expand_as(w) ) + (c * (a[0].expand_as(c) + b[0].expand_as(c)))
#         # x = c
#         # x = x + self.drop_path( h * (a[0].expand_as(h) ))
#         # x = x + self.drop_path( w * ( b[0].expand_as(w)))
#         # x = x + self.drop_path(c * (a[0].expand_as(c) + b[0].expand_as(c)))
#
#
#         x = (x * (h + w)).permute(0, 3, 2, 1)
#
#         return x+res


import matplotlib.pyplot as plt
import numpy.fft as nf
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act1 = act_layer()
#         self.act2 = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act1(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.act2(x)
#         x = self.drop(x)
#         return x
#
#
# class WeightedPermuteMLP(nn.Module):
#     def __init__(self, dim, c_dim, len_sequence, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#                  proj_drop=0.):
#         super().__init__()
#         self.segment_dim = segment_dim
#         self.T = len_sequence
#         self.ratio = (self.T - 1) / 2
#         self.mlp_c = nn.Linear(int(c_dim), c_dim, bias=qkv_bias)
#         self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
#         self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
#
#         self.v1_h = nn.Linear(c_dim, c_dim, bias=qkv_bias)
#         self.v1_w = nn.Linear(c_dim, c_dim, bias=qkv_bias)
#
#         self.reweight_h = Mlp(c_dim, c_dim // 4, c_dim )
#         self.reweight_w = Mlp(c_dim, c_dim // 4, c_dim )
#         self.reweight = Mlp(c_dim, c_dim // 4, c_dim * 3)
#
#         self.proj = nn.Linear(c_dim, c_dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.norm1 = nn.LayerNorm(c_dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=int(c_dim * self.ratio), hidden_features=c_dim // 4, out_features=c_dim,act_layer=nn.ReLU)
#         self.drop_path = DropPath(0.1)
#
#
#         self.T_extraction_h = nn.Conv2d(in_channels=self.segment_dim, out_channels=self.segment_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=self.segment_dim)
#         self.T_extraction_w = nn.Conv2d(in_channels=self.segment_dim, out_channels=self.segment_dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
#         self.sigm = nn.Tanh()
#     def multi_diff(self, data):
#         B, H, W, C = data.shape
#         data = data.reshape(B * C // self.T, H, W, self.T)
#         hid = torch.zeros(B * C // self.T, H, W, 1).to('cuda')
#         # data = torch.cat([data[:,0].unsqueeze(1), data], dim=1)
#         for i in range(0, 10):
#             hid = torch.cat([hid, data[:, :, :, 0: self.T - i - 1] - data[:, :, :, i + 1:]], dim=3)
#
#         return hid[:, :, :, 1:].reshape(B, H, W, -1)
#
#     def fft_trans(self, T, sr):
#         complex_ary = nf.fft(sr)
#         y_ = nf.ifft(complex_ary).real
#         fft_freq= nf.fftfreq(y_.size, T[1]-T[0])
#         fft_pow = np.abs(complex_ary)
#         return fft_freq, fft_pow
#
#     def forward(self, x):
#
#         x = x.permute(0, 3, 2, 1)
#         B, H, W, C = x.shape
#         x = self.norm1(x)
#         # img_datax = x.cpu().detach().numpy()
#         # imgx = img_datax[0,0,:,:]
#         # curl = imgx[0, :]
#         # plt.subplot(2, 1, 1)
#         # plt.plot(curl)
#         #
#         # data = imgx[0, :]
#         # t = np.linspace(0,len(data)-1,len(data))
#         # l1, l2 = self.fft_trans(t, data)
#         # plt.subplot(2, 1, 2)
#         # plt.plot(l1[l1 > 0], l2[l1 > 0]/2, '-', lw=2)
#         # plt.show()
#
#         S = C // self.segment_dim
#         h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H * S)
#         # h = self.T_extraction_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
#
#
#         h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
#
#         w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W * S)
#         # w = self.T_extraction_w(w)
#         w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)
#
#         c = self.mlp_c(x)
#
#         a = (h).mean(1)
#         b = (w).mean(2)
#         a = self.reweight_h(a ).reshape(B, H, C, 1).permute(3, 0, 1, 2).unsqueeze(2)
#         b = self.reweight_w(b ).reshape(B, W, C, 1).permute(3, 0, 1, 2).unsqueeze(3)
#         # a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).unsqueeze(2).unsqueeze(2)
#
#         x = h * (a[0].expand_as(h) ) + w * ( b[0].expand_as(w) ) + self.drop_path(c * (a[0].expand_as(c) + b[0].expand_as(c)))
#         # x = c
#         # x = x + self.drop_path( h * (a[0].expand_as(h) ))
#         # x = x + self.drop_path( w * ( b[0].expand_as(w)))
#         # x = x + self.drop_path(c * (a[0].expand_as(c) + b[0].expand_as(c)))
#         x = (x).permute(0, 3, 2, 1)
#
#         return x

class PositionalEncoding(nn.Module):
    """位置编码"""
    #num+hiddens:向量长度  max_len:序列最大长度
    def __init__(self, num_hiddens, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P : (1, 1000, 32)
        self.P = torch.zeros((1, max_len, num_hiddens))
        #本例中X的维度为(1000, 16)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)   #::2意为指定步长为2 为[start_index : end_index : step]省略end_index的写法
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act1 = act_layer()
#         self.act2 = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act1(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.act2(x)
#         x = self.drop(x)
#         return x
#
# class Mlp_anti_act(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act1 = act_layer()
#         self.act2 = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act1(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.act2(x)
#         x = self.drop(x)
#         return x
#
#
# class WeightedPermuteMLP(nn.Module):
#     def __init__(self, dim, c_dim, len_sequence, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#                  proj_drop=0.):
#         super().__init__()
#         self.segment_dim = segment_dim
#         self.T = len_sequence
#         self.size = dim // len_sequence
#         self.ratio = (self.T - 1) / 2
#         self.mlp_c = Mlp(c_dim, c_dim // 4, c_dim )
#         self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
#         self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
#
#         self.v1_h = nn.Linear(c_dim, c_dim, bias=qkv_bias)
#         self.v1_w = nn.Linear(c_dim, c_dim, bias=qkv_bias)
#
#         self.reweight_h = Mlp(c_dim, c_dim // 4, c_dim )
#         self.reweight_w = Mlp(c_dim, c_dim // 4, c_dim )
#         self.reweight = Mlp(c_dim, c_dim // 4, c_dim * 3)
#
#         self.proj = nn.Linear(c_dim, c_dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.norm1 = nn.LayerNorm(c_dim)
#         self.norm2 = nn.LayerNorm(c_dim)
#         self.norm3 = nn.LayerNorm(c_dim)
#         self.mlp = Mlp(in_features=int(c_dim * self.ratio), hidden_features=c_dim // 4, out_features=c_dim,act_layer=nn.ReLU)
#
#         self.HBC_fast_on_cell = nn.Linear(dim, dim//2 )
#         self.HBC_fast_off_cell =  nn.Linear(dim, dim//2 )
#
#         self.HBC_slow_on_cell = nn.Conv1d(in_channels = self.size//8 ,out_channels= self.size//8 , kernel_size=10, stride=2, padding=9, dilation=2)
#         self.HBC_slow_off_cell = nn.Conv1d(in_channels = self.size//8 ,out_channels= self.size//8 , kernel_size=10, stride=2, padding=9, dilation=2)
#
#         self.test = nn.Conv1d(in_channels = self.size * self.segment_dim,out_channels= self.size * self.segment_dim, kernel_size=9, stride=2, padding=8, dilation=2)
#         self.pos_encoding = PositionalEncoding(64 * self.size, 0)
#
#         self.HSAC_on = Mlp(dim, 2 * dim // 4, dim )
#         self.HSAC_off = Mlp_anti_act(dim, 2 * dim // 4, dim)
#
#         self.Hfuse = Mlp(2 * dim, dim // 4, dim)
#
#
#         self.WBC_fast_on_cell = nn.Linear(dim, dim//2 )
#         self.WBC_fast_off_cell = nn.Linear(dim, dim//2 )
#         self.WBC_slow_on_cell = nn.Conv1d(in_channels = self.size ,out_channels= self.size , kernel_size=9, stride=2, padding=8, dilation=2)
#         self.WBC_slow_off_cell = nn.Conv1d(in_channels = self.size ,out_channels= self.size , kernel_size=9, stride=2, padding=8, dilation=2)
#
#         self.WSAC_on = Mlp(dim, 2 * dim // 4, dim )
#         self.WSAC_off = Mlp_anti_act(dim, 2 * dim // 4, dim)
#
#         self.Wfuse = Mlp(2 * dim, dim // 4, dim)
#
#         self.act1 = nn.ReLU()
#         self.act2 = nn.ReLU()
#         self.act3 = nn.ReLU()
#
#     def forward(self, x):
#         x = x.permute(0, 3, 2, 1)
#
#         B, H, W, C = x.shape
#
#         x = self.norm1(x)
#
#         # S = 10
#         # self.segment_dim = C//S
#         S = C // self.segment_dim
#         h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H * S)
#         # h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
#         # h = self.pos_encoding(h.reshape(B,-1,H * S).permute(0,2,1)).permute(0,2,1).reshape(B, self.segment_dim, W, H * S)
#         BC_fast_on = self.HBC_fast_on_cell(h)  #         (B, self.segment_dim, W, H * S)
#         BC_fast_off = self.HBC_fast_off_cell(h) # -ReLU  (B, self.segment_dim, W, H * S)
#
#         BC_slow_on = self.HBC_slow_on_cell(h.permute(0, 2, 1, 3).reshape(16,-1,H*S*self.segment_dim)).reshape(B, W, self.segment_dim, -1) #          (B, self.segment_dim, W, H * S)
#         BC_slow_off = self.HBC_slow_off_cell(h.permute(0, 2, 1, 3).reshape(16,-1,H*S*self.segment_dim)).reshape(B,  W, self.segment_dim, -1) # -ReLU  (B, self.segment_dim, W, H * S)
#
#         BC_slow_on = BC_slow_on.permute(0,2,1,3)
#         BC_slow_off = BC_slow_off.permute(0, 2, 1, 3)
#
#         SAC_on_out = self.HSAC_on(torch.cat((BC_fast_on, BC_slow_on), dim=-1))  # ReLU     (B, self.segment_dim, W,  H * S)
#
#         SAC_off_out = self.HSAC_off(torch.cat((BC_fast_off, BC_slow_off), dim=-1)) # -ReLU  (B, self.segment_dim, W, H * S)
#
#         Hx = self.Hfuse(torch.cat((SAC_on_out, SAC_off_out), dim=-1)).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
#         Hx = self.act3(Hx)
#         Hx = self.norm2(Hx)
#
#
#         # w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W * S)
#         # WBC_fast_on = self.WBC_fast_on_cell(w)  #         (B, self.segment_dim, W, H * S)
#         # WBC_fast_off = self.WBC_fast_on_cell(w) # -ReLU  (B, self.segment_dim, W, H * S)
#         #
#         # WBC_slow_on = self.WBC_slow_on_cell(w.reshape(16,-1,W*10*self.segment_dim)).reshape(B, H, self.segment_dim, -1) #          (B, self.segment_dim, W, H * S)
#         # WBC_slow_off = self.WBC_slow_off_cell(w.reshape(16,-1,W*10*self.segment_dim)).reshape(B,  H, self.segment_dim, -1) # -ReLU  (B, self.segment_dim, W, H * S)
#         #
#         #
#         # WSAC_on_out = self.WSAC_on(torch.cat((WBC_fast_on, WBC_slow_on), dim=-1))  # ReLU     (B, self.segment_dim, W,  H * S)
#         #
#         # WSAC_off_out = self.HSAC_off(torch.cat((WBC_fast_off, WBC_slow_off), dim=-1)) # -ReLU  (B, self.segment_dim, W, H * S)
#         #
#         # Wx = self.Wfuse(torch.cat((WSAC_on_out, WSAC_off_out), dim=-1)).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)
#         # Wx = self.act2(Wx)
#         # Wx = self.norm3(Wx)
#
#         c = self.mlp_c(x)
#
#         x = c * Hx
#         # x = c
#         x = (x).permute(0, 3, 2, 1)
#
#         return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hid, kernel, kernel2, figure_size):
        super().__init__()
        kernel2 = 7
        self.T = 4
        self.hid = hid
        # self.inv1 = My_involution(channels_in=self.T,
        #                       channels_out=1,
        #                       kernel_size=kernel,
        #                       stride=1,
        #                       padding=(kernel-1)//2,
        #                       group_channel=1)
        inchannel = 0
        for i in range(self.T):
            inchannel = inchannel + i
        self.conv1 = nn.Conv2d(
            in_channels=inchannel,
            out_channels=self.T,
            kernel_size=kernel2,
            stride=1,
            padding=(kernel2-1)//2,
            bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=self.hid * self.T * 2,
            out_channels=self.hid * self.T,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        reduction = 2
        self.linear1 = nn.Sequential(
            nn.Linear(self.T, self.T // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(self.T // reduction, self.T, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.Sigmoid()
        self.bn3 = nn.BatchNorm2d(num_features=1)

        self.SE1 = torch.nn.AdaptiveAvgPool2d(1)

        self.inv = involution(self.hid * self.T, kernel, 1)


    def forward(self, data):
        B, C, H, W = data.shape
        latent = self.inv(data)
        # latent = data
        data = data.view(B * C//self.T, self.T, H, W)

        hid = torch.zeros(B * C//self.T, 1, H, W).to('cuda')
        # data = torch.cat([data[:,0].unsqueeze(1), data], dim=1)
        for i in range(0, self.T):
            # _, dif1 = data.split([i + 1, 10], dim=1)
            # _, dif2 = data.split([0, self.T - i], dim=1)
            dif1 = data[:, i + 1: self.T]
            dif2 = data[:, 0: self.T - i - 1]
            hid = torch.cat([hid, dif2-dif1], dim=1)
        hid = self.conv1(hid[:, 1:])
        data = hid.view(B, C, H, W) + latent

        # data = test1# + latent
        # cat_data = torch.cat([hid.view(B, C, H, W), latent], dim=1)
        # data = self.conv2(cat_data)
        return data
#

        #     hid1 = self.conv1((data[:, i: i+10] - data[:, i-1: i+9]))
        #     hid = torch.cat([hid, hid1],dim=1)
        #     # o = (torch.mul(hid1, data[:, -1].unsqueeze(1)+1))
        #     # hid = self.relu((o-1))
        #
        #     # o = torch.mul(hid1, data[:, -1].unsqueeze(1)+1)
        #     o=hid1+data[:, -1].unsqueeze(1)
        #     o = self.relu((o-1))
        #     # o = self.conv2(torch.cat([hid1, data[:, -1].unsqueeze(1)], dim=1))
        #     data = torch.cat([data, o], dim=1) # B, hid, T+1
        # C_se = self.linear2(self.linear1(self.SE1(hid[:, 1: self.T + 1]).view(B * C // self.T, self.T)))  # [16,80,1,1]
        # C_se = C_se.view(B * C // self.T, self.T, 1, 1).expand_as(data[:, self.T: self.T+10])  # [16*16,10,64,64]
        # result = C_se * data[:, self.T: self.T+10]
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, hid, kernel,figure_size):
#         super().__init__()
#         batch = 16
#         self.K = 7
#         self.G = 1
#         self.scale = 1
#         self.hid_layer = hid
#         self.conv = nn.Conv2d(in_channels=hid * 10, out_channels=hid * 10//4, kernel_size=7, stride=1, padding=3)
#         self.conv1 = nn.Conv2d(in_channels=hid * 10//4, out_channels=hid * 10, kernel_size=7, stride=1, padding=3)
#
#         self.unfold = nn.Unfold(self.K, 1, (self.K-1)//2, 1)
#
#         self.reduce = nn.Conv2d(hid * 10, hid * 10 // 4, 1)
#         self.span = nn.Conv2d(hid * 10 // 4, self.K * self.K * self.G, 1)
#
#
#         self.inv = involution(hid * 10, kernel, 1)
#
#         self.Wqh = nn.Parameter(torch.randn((batch, hid * 10, self.K, figure_size), device='cuda'))
#         self.Wqw = nn.Parameter(torch.randn((batch, hid * 10, figure_size, self.K), device='cuda'))
#
#         self.Wkh = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#         self.Wkw = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#
#         # self.Wvh = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#         # self.Wvw = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#
#         self.bias = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#         self.SE1 = torch.nn.AdaptiveAvgPool2d(1)
#         self.SE2 = torch.nn.AdaptiveAvgPool2d(1)
#         self.SE3 = torch.nn.AdaptiveAvgPool2d(1)
#
#         self.norm = nn.GroupNorm(2, hid*10)
#         self.relu = nn.ReLU(inplace=True)
#
#
#     def forward(self, data, mask=None):
#         B, C, H, W = data.shape
#         #hid = self.inv(data) # # [16,80,64,64]
#
#         # 计算q,k,v
#         # data [16,80,64,64]
#         q = torch.matmul(torch.matmul(self.Wqh, data),self.Wqw) #[16,80,64,64]
#         q = q.view(B, C, self.K*self.K, 1, 1)
#         q = q.expand(B, C, self.K*self.K, H, W)
#         # k = torch.mul(torch.mul(self.Wkh, data),self.Wkw)
#
#         # y = self.conv1(self.conv(torch.cat((q,k), dim=1))) ##[16,80,64,64]
#         y = self.conv1(self.conv(data))  ##[16,80,64,64]
#         x_unfolded = self.unfold(y).view(B, C, self.K * self.K, H, W)
#         out = torch.mul(x_unfolded, q).sum(dim=2)
#         # kernel = self.span(self.reduce(q)) # B,KxKxG,H,W
#         # kernel = kernel.view(B, self.G, self.K * self.K, H, W).unsqueeze(2)
#         # #
#         # x_unfolded = self.unfold(y)
#         # x_unfolded = x_unfolded.view(B, self.G, C // self.G, self.K * self.K, H, W)
#         # out = torch.mul(kernel, x_unfolded).sum(dim=3)  # B,G,C/G,H,W
#         out = out.view(B, C, H, W)
#
#         return out

# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, hid, kernel,figure_size):
#         super().__init__()
#         batch = 8
#         self.scale = 1
#         self.hid_layer = hid
#         self.conv = nn.Conv2d(in_channels=hid * 10, out_channels=hid * 30, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=hid * 10, out_channels=hid * 10, kernel_size=kernel, stride=1, padding=(kernel-1)//2)
#         self.inv = involution(hid * 10, kernel, 1)
#         self.Wqh = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#         self.Wqw = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#
#         self.Wkh = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#         self.Wkw = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#
#         self.Wvh = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#         self.Wvw = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#
#         self.bias = nn.Parameter(torch.randn((batch, hid * 10, figure_size, figure_size), device='cuda'))
#
#
#         self.SE1 = torch.nn.AdaptiveAvgPool2d(1)
#         self.SE2 = torch.nn.AdaptiveAvgPool2d(1)
#         self.SE3 = torch.nn.AdaptiveAvgPool2d(1)
#         self.norm = nn.BatchNorm2d(hid,affine=True)
#         self.act = nn.LeakyReLU(0.2, inplace=True)
#         reduction = 8
#         self.linear1 = nn.Sequential(
#             nn.Linear(hid * 10, hid * 10 // reduction, bias=False),
#             nn.ReLU(inplace=True)
#         )
#         self.linear2 = nn.Sequential(
#             nn.Linear(hid * 10 // reduction, hid * 10, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, data, mask=None):
#         B, C, H, W = data.shape
#
#         # hid = self.inv(data) # # [16,80,64,64]
#         # 计算q,k,v
#         # data [16,80,64,64]
#         q = torch.mul(torch.mul(self.Wqh, data),self.Wqw) #[16,80,64,64]
#         k = torch.mul(torch.mul(self.Wkh, data),self.Wkw)
#         v = torch.mul(torch.mul(self.Wvh, data),self.Wvw)
#
#         kt = k.permute(0,1,3,2)
#         kt = kt.contiguous().view(B*self.hid_layer, 10*W, H)
#         q = q.view(B*self.hid_layer, H, 10*W)
#         a = torch.bmm(kt, q)/(10*W)
#         hb, Hp, Wp = a.shape
#         a = (self.norm(a.view(B, self.hid_layer, Hp, Wp)))
#         score = a.view(hb, Hp, Wp)
#         v = v.view(B*self.hid_layer, H, 10*W)
#         y = torch.bmm(v, score).view(B, C, H, W)
#
#         # C_se = self.linear2(self.linear1(self.SE1(a).view(B, self.hid_layer * 10)))  # [16,80,1,1]
#         # C_se = C_se.view(B, self.hid_layer * 10, 1, 1)
#         # # H_se = self.SE1(a.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # [16,1,64,1]
#         # # W_se = self.SE1(a.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # [16,1,1,64]
#         #
#         # C_se = C_se.expand_as(q) # [16,80,64,64]
#         # # H_se = H_se.expand_as(v)  # [16,80,64,64]
#         # # W_se = W_se.expand_as(v)  # [16,80,64,64]
#         # result = hid #* C_se #* H_se * W_se
#         return y

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 feature_count,
                 in_channels,
                 out_channels,
                 num_block,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.num_block = num_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.feature_count = feature_count

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            padding=0,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        # *******************************************************************


        if self.feature_count == 0:
            # self.conv2 = involution(self.mid_channels, 1, self.conv2_stride)
            # self.atten = ScaledDotProductAttention(32,1, 7, 32)
            seg_dim = 4
            input_size = 64
            dim = self.mid_channels * input_size // seg_dim
            self.atten = WeightedPermuteMLP(dim=dim, c_dim= self.mid_channels, len_sequence = 10, segment_dim=seg_dim, input_size=input_size)

            seg_dim = 8
            input_size = 64
            dim = self.mid_channels * input_size // seg_dim
            self.conv2 = WeightedPermuteMLP(dim=dim, c_dim= self.mid_channels, len_sequence = 10, segment_dim=seg_dim, input_size=input_size)
        elif self.feature_count == 1:
            # self.conv2 = involution(self.mid_channels, 15, self.conv2_stride)

            # self.atten = ScaledDotProductAttention(32,15, 7, 16)
            seg_dim = 4
            input_size = 32
            dim = self.mid_channels * input_size // seg_dim
            self.atten = WeightedPermuteMLP(dim=dim, c_dim= self.mid_channels,len_sequence = 10,segment_dim=seg_dim, input_size=input_size)

            seg_dim = 8
            input_size = 32
            dim = self.mid_channels * input_size // seg_dim
            self.conv2 = WeightedPermuteMLP(dim=dim, c_dim= self.mid_channels, len_sequence = 10, segment_dim=seg_dim, input_size=input_size)

            # self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, stride=1)
            # self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=7, padding=(7-1)//2, stride=1)
        elif self.feature_count == 2:
            # self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=15, padding=(15-1)//2, stride=1)
            # self.atten = ScaledDotProductAttention(32,15, 7, 16)
            seg_dim = 4
            input_size = 32
            dim = self.mid_channels * input_size // seg_dim
            self.atten = WeightedPermuteMLP(dim=dim, c_dim= self.mid_channels,len_sequence = 10,segment_dim=seg_dim, input_size=input_size)

            seg_dim = 8
            input_size = 32
            dim = self.mid_channels * input_size // seg_dim
            self.conv2 = WeightedPermuteMLP(dim=dim, c_dim= self.mid_channels, len_sequence = 10, segment_dim=seg_dim, input_size=input_size)
            # self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, stride=1)
            # self.conv2 = involution(self.mid_channels, 15, self.conv2_stride)
        else:
            # self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=21, padding=(21-1)//2, stride=1)
            # self.atten = ScaledDotProductAttention(32,21, 9, 8)
            seg_dim = 4
            input_size = 16
            dim = self.mid_channels * input_size // seg_dim
            self.atten = WeightedPermuteMLP(dim=dim, c_dim= self.mid_channels,len_sequence = 10, segment_dim=seg_dim, input_size=input_size)

            seg_dim = 8
            input_size = 16
            dim = self.mid_channels * input_size // seg_dim
            self.conv2 = WeightedPermuteMLP(dim=dim, c_dim= self.mid_channels, len_sequence = 10, segment_dim=seg_dim, input_size=input_size)
            # self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, stride=1)
            # self.conv2 = involution(self.mid_channels, 21, self.conv2_stride)
        # self.me = MEModule(16)

        self.dropout = nn.Dropout(p=0.5)
        # *******************************************************************
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels * 2,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.conv4 = nn.Conv2d(in_channels=self.mid_channels-1, out_channels=self.mid_channels, kernel_size=1, bias=False)

        self.sig = nn.Sigmoid()


    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    # def forward(self, x):
    #
    #     def _inner_forward(x):
    #         identity = x
    #         out = self.conv1(x)
    #         out = self.norm1(out)
    #         out = self.relu(out)
    #
    #         # out1 = self.dw1(out)
    #         # out1 = self.dw2(out1)
    #         out2 = self.atten(out)
    #         # out2 = self.atten1(out2)
    #         out1 = self.conv2(out)
    #         # out1 = self.dw1_norm(out1)
    #         # out1 = self.dw1_act(out1)
    #         # out1 = self.dw2(out1)
    #
    #         out3 = torch.diff(out, n=1, dim=1)
    #         out3 = self.conv4(out3)
    #         out3 = self.sig((out3))
    #
    #         out = self.conv3(torch.cat((out2, out1*out3),dim=1))
    #         out = self.norm3(out)
    #         # out = self.relu(out)
    #
    #         if self.downsample is not None:
    #             identity = self.downsample(x)
    #         if self.num_block >=4:
    #             out += identity
    #
    #         return out
    #
    #     if self.with_cp and x.requires_grad:
    #         out = cp.checkpoint(_inner_forward, x)
    #     else:
    #         out = _inner_forward(x)
    #     out = self.relu(out)
    #     return out

    def forward(self, x):


        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        # out1 = self.dw1(out)
        # out1 = self.dw2(out1)
        out2 = self.atten(out)
        # out2 = self.atten1(out2)
        out1 = self.conv2(out)
        # out1 = self.dw1_norm(out1)
        # out1 = self.dw1_act(out1)
        # out1 = self.dw2(out1)

        # out3 = torch.diff(out, n=1, dim=1)
        # out3 = self.conv4(out3)
        # out3 = self.sig((out3))

        out = self.conv3(torch.cat((out1, out2),dim=1))
        # out = self.conv3(out1 + torch.abs(out3) * out2)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if self.num_block >=4:
            out = out + identity

        out = self.relu(out)



        return out



def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError('expansion is not specified for {}'.format(block.__name__))
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 feature_count,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)
        self.feature_count = feature_count
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                feature_count=self.feature_count,
                in_channels=in_channels,
                out_channels=out_channels,
                num_block = num_blocks,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    feature_count=self.feature_count,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_block = num_blocks,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        #layers.append(BottleneckBlock(in_channels=in_channels, fmap_size=(16, 16), out_channels= in_channels,heads=4))
        super(ResLayer, self).__init__(*layers)


@BACKBONES.register_module()
class RedNet(BaseBackbone):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`_ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        # >>> from mmcls.models import ResNet
        # >>> import torch
        # >>> self = ResNet(depth=18)
        # >>> self.eval()
        # >>> inputs = torch.rand(1, 3, 32, 32)
        # >>> level_outputs = self.forward(inputs)
        # >>> for level_out in level_outputs:
        # ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        26: (Bottleneck, (1, 8, 2, 8)),
        38: (Bottleneck, (1, 1, 1, 1)),
        50: (Bottleneck, (9, 9, 3, 2)),
        101: (Bottleneck, (2, 4, 2, 4)),
        152: (Bottleneck, (2, 8, 2, 8))
    }
    # 单[5,8]的mse效果和[2,8,2,8]差不多，但是mae很大，mae反应的是整体的情况，mse很大是少部分相差特别大
    # 所以深层的enc可以对整体有提升，mae可以很小，能够到1300多，但是mse还是很大，可以试下把浅层的深度加深或者用其他手段来提升mse。
    def __init__(self,
                 depth,
                 layer_config,
                 in_channels=640,
                 stem_channels=640,
                 base_channels=160,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True):
        super(RedNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        # assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        # assert max(out_indices) < num_stages
        self.style = style
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        stage_blocks = layer_config
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels , stem_channels )
        self.in_channels = in_channels
        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion#base_channels=64
        for i, num_blocks in enumerate(self.stage_blocks):  #self.stage_blocks最多等于4
            stride = strides[i]
            dilation = dilations[i]
            #
            # if i == 0:
            #     # _in_channels = in_channels
            #     _out_channels = _in_channels
            # elif i == 1:
            #     # _in_channels = in_channels
            #     _out_channels = _in_channels
            # elif i == 2:
            #     # _in_channels = in_channels
            #     _out_channels = _in_channels
            # else:
            #     # _in_channels = in_channels
            #     _out_channels = _in_channels
            if num_blocks == 0:
                res_layer = nn.Identity()
            else:
                res_layer = self.make_res_layer(
                    feature_count = i,
                    block=self.block,
                    num_blocks=num_blocks,
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg)
            #_in_channels = _out_channels

            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
        #self.feat_dim = res_layer[-1].out_channels
        # self.apply(self.init_weights)

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.stem = nn.Sequential(
            ConvModule(
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            ConvModule(
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        # _, c, h, w = enc1.shape  #B*T, hid_s, h, w
        # enc1 = enc1.view(-1,self.in_channels, h, w)
        # _, c, h, w = enc2.shape
        # enc2 = enc2.view(-1,self.in_channels, h, w)
        # _, c, h, w = enc3.shape
        # enc3 = enc3.view(-1,self.in_channels, h, w)

        # x = self.stem(x)
        # x = self.maxpool(x)
        # enc1 = x[0]
        # enc2 = x[1]
        # enc3 = x[2]
        # enc4 = x[3]
        x[-1] = self.stem(x[-1])
        x[-1] = self.maxpool(x[-1])
        outs = []

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            # if i == 0:##第一个stage
            #     enc1 = res_layer(enc1)
            #     outs.append(enc1)
            # elif i == 1: #第二个stage
            #     enc2 = res_layer(enc2)
            #     outs.append(enc2)
            # elif i ==2:
            #     enc3 = res_layer(enc3)  # 把3和4的num_blocks位置调换一下
            #     outs.append(enc3)
            # else:
            #     enc3 = res_layer(enc3)
            #     outs.append(enc4)
            # outs.append(self.stem(x[i]))
            # outs[i]=self.maxpool(outs[i])
            outs.append(res_layer(x[i]))

            # if i == 3:
            #     outs.append(res_layer(x[i]))
            # else:
            #     outs.append(x[i])
        return outs

    def train(self, mode=True):
        super(RedNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
