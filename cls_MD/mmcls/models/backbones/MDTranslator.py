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


class MDA_MLP(nn.Module):
    def __init__(self, dim, c_dim, len_sequence, segment_dim=8, input_size=64, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim
        self.T = len_sequence
        self.ratio = (self.T - 1) / 2
        self.mlp_c = nn.Linear(c_dim, c_dim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

        self.MLP_h = Mlp(c_dim, c_dim // 4, c_dim)
        self.MLP_w = Mlp(c_dim, c_dim // 4, c_dim)

        self.norm1 = nn.LayerNorm(c_dim)
        self.norm2 = nn.LayerNorm(c_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = x.permute(0, 3, 2, 1)

        B, H, W, C = x.shape

        x = self.norm1(x)

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim * W, H * S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H * self.segment_dim, W * S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)

        a = (h).mean(1)
        b = (w).mean(2)

        a = self.MLP_h(a).reshape(B, 1, W, C)
        b = self.MLP_w(b).reshape(B, H, 1, C)

        x = h * (b.expand_as(h)) + w * ( a.expand_as(w)) + (c * (a + b))

        x = (x).permute(0, 3, 2, 1)

        return x


class MDAUnit(nn.Module):
    """MDAUnit.

    Args:
        feature_count (int): layer_count in MDATranslator.
        in_channels (int): input channels of this MDAUnit.
        out_channels (int): output channels of this MDAUnit.
        num_block (int): number of MDAUnit in this layer
        data_size (int): input datasize
        g1 (int): group number 1
        g1 (int): group number 2
        reduction (int): channel reduction grammar
    Input:
        X: (B,C,H,W)
    Output:
        out: (B,C,H,W)
    """

    def __init__(self,
                 feature_count,
                 in_channels,
                 out_channels,
                 num_block,
                 data_size,
                 g1,
                 g2,
                 reduction=4,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(MDAUnit, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.num_block = num_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        assert out_channels % reduction == 0
        self.mid_channels = out_channels // reduction

        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.feature_count = feature_count


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
            stride=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        # *******************************************************************


        if self.feature_count == 0:

            seg_dim = g1
            input_size = data_size
            dim = self.mid_channels * input_size // seg_dim
            self.atten = MDA_MLP(dim=dim, c_dim= self.mid_channels, len_sequence = 10, segment_dim=seg_dim, input_size=input_size)

            seg_dim = g2
            input_size = data_size
            dim = self.mid_channels * input_size // seg_dim
            self.conv2 = MDA_MLP(dim=dim, c_dim= self.mid_channels, len_sequence = 10, segment_dim=seg_dim, input_size=input_size)
        elif self.feature_count == 1:

            seg_dim = g1
            input_size = data_size // 2
            dim = self.mid_channels * input_size // seg_dim
            self.atten = MDA_MLP(dim=dim, c_dim= self.mid_channels,len_sequence = 10,segment_dim=seg_dim, input_size=input_size)

            seg_dim = g2
            input_size = data_size // 2
            dim = self.mid_channels * input_size // seg_dim
            self.conv2 = MDA_MLP(dim=dim, c_dim= self.mid_channels, len_sequence = 10, segment_dim=seg_dim, input_size=input_size)

        elif self.feature_count == 2:

            seg_dim = g1
            input_size = data_size // 2
            dim = self.mid_channels * input_size // seg_dim
            self.atten = MDA_MLP(dim=dim, c_dim= self.mid_channels,len_sequence = 10,segment_dim=seg_dim, input_size=input_size)

            seg_dim = g2
            input_size = data_size // 2
            dim = self.mid_channels * input_size // seg_dim
            self.conv2 = MDA_MLP(dim=dim, c_dim= self.mid_channels, len_sequence = 10, segment_dim=seg_dim, input_size=input_size)

        else:

            seg_dim = g1
            input_size = data_size // 4
            dim = self.mid_channels * input_size // seg_dim
            self.atten = MDA_MLP(dim=dim, c_dim= self.mid_channels,len_sequence = 10, segment_dim=seg_dim, input_size=input_size)

            seg_dim = g2
            input_size = data_size // 4
            dim = self.mid_channels * input_size // seg_dim
            self.conv2 = MDA_MLP(dim=dim, c_dim= self.mid_channels, len_sequence = 10, segment_dim=seg_dim, input_size=input_size)


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


    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)


    def forward(self, x):

        ###    MDAUnit    ####
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out2 = self.atten(out)
        out1 = self.conv2(out)


        out = self.conv3(torch.cat((out1, out2),dim=1))

        out = self.norm3(out)

        if self.num_block >=4:
            out = out + identity

        out = self.relu(out)

        return out




class MDALayer(nn.Sequential):
    """MDALayer to build MDATranslator.

    Args:
        feature_count (int): layer count.
        block (nn.Module): temporal unit in this layer, here is the MDAUnit.
        num_blocks (int): number of MDAUnit.
        in_channels (int): input channels of this MDAUnit.
        out_channels (int): output channels of this MDAUnit.
        num_block (int): number of MDAUnit in this layer
        data_size (int): input datasize
        g1 (int): group number 1
        g1 (int): group number 2
        reduction (int): channel reduction grammar
    """

    def __init__(self,
                 feature_count,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 data_size,
                 g1,
                 g2,
                 reduction=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        self.block = block
        self.reduction = reduction
        self.feature_count = feature_count

        layers = []
        layers.append(
            block(
                feature_count=self.feature_count,
                in_channels=in_channels,
                out_channels=out_channels,
                num_block = num_blocks,
                data_size=data_size,
                g1=g1,
                g2=g2,
                reduction=self.reduction,
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
                    data_size=data_size,
                    g1=g1,
                    g2=g2,
                    reduction=self.reduction,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))

        super(MDALayer, self).__init__(*layers)


@BACKBONES.register_module()
class MDATranslator(BaseBackbone):
    """MDATranslator.

    Args:
        layers (int): layer num of MDATranslator
        layer_config (list): MDAUnit number of each layer
        reduction (int): Reduction rate 'grammar'
        data_size (int): input datasize
        g1 (int): group number 1
        g1 (int): group number 2
    Input:
        X: list: [tensor1, tensor2, ..., tensor L]
    Output:
        X: list: [tensor1, tensor2, ..., tensor L]

    Example:
        # >>> from mmcls.models import MDATranslator
        # >>> import torch
        # >>> self = MDATranslator(layers=4, layer_config=[1,1,1,1], in_channels=160, reduction=2, data_size=64, g1=2, g2=4)
        # >>> self.eval()
        # >>> inputs = [torch.rand(1, 160, 64, 64) for i in range(0,4)]
        # >>> outputs = self.forward(inputs)
        # >>> print(outputs)
    """

    def __init__(self,
                 layers,
                 layer_config,
                 in_channels=640,
                 reduction=2,
                 data_size=64,
                 g1=2,
                 g2=4,
                 out_indices=(3, ),
                 style='pytorch',
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True):
        super(MDATranslator, self).__init__()


        self.num_stages = layers
        assert layers >= 1 and layers <= 4
        self.out_indices = out_indices
        self.style = style
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        stage_blocks = layer_config
        self.block = MDAUnit
        self.stage_blocks = stage_blocks[:layers]
        self.reduction = reduction

        self._make_stem_layer(in_channels , in_channels )
        self.in_channels = in_channels
        self.res_layers = []
        # _in_channels = in_channels
        # _out_channels = in_channels
        for i, num_blocks in enumerate(self.stage_blocks):  #self.stage_blocks最多等于4
            if num_blocks == 0:
                res_layer = nn.Identity()
            else:
                res_layer = self.make_mda_layer(
                    feature_count = i,
                    block=self.block,
                    num_blocks=num_blocks,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    reduction=self.reduction,
                    data_size=data_size,
                    g1=g1,
                    g2=g2,
                    style=self.style,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg)
            #_in_channels = _out_channels

            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()


    def make_mda_layer(self, **kwargs):
        return MDALayer(**kwargs)

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

        x[-1] = self.stem(x[-1])
        x[-1] = self.maxpool(x[-1])
        outs = []

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            outs.append(res_layer(x[i]))
        return outs

    def train(self, mode=True):
        super(MDATranslator, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
