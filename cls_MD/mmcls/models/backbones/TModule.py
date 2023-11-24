import torch.nn as nn
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
import numpy as np
import torch
from timm.models.layers import DropPath, trunc_normal_
import math


def softmax(x):
    e_x = np.exp((x-torch.max(x)).detach().numpy())# 防溢出
    return e_x/e_x.sum(0)

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
    def __init__(self, dim, c_dim, segment_dim=8, bias=False):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c = nn.Linear(c_dim, c_dim, bias=bias)
        self.mlp_h = nn.Linear(dim, dim, bias=bias)
        self.mlp_w = nn.Linear(dim, dim, bias=bias)

        self.MLP_h = Mlp(c_dim, c_dim // 4, c_dim)
        self.MLP_w = Mlp(c_dim, c_dim // 4, c_dim)

        self.norm1 = nn.LayerNorm(c_dim)
        self.norm2 = nn.LayerNorm(c_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # input: B, C, H, W
        x = x.permute(0, 3, 2, 1)
        # B, H, W, C
        B, H, W, C = x.shape

        x = self.norm1(x)

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim * W, H * S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H * self.segment_dim, W * S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)

        a = h.mean(1)  # average on H-dimension
        b = w.mean(2)  # average on W-dimension

        a = self.MLP_h(a).reshape(B, 1, W, C)
        b = self.MLP_w(b).reshape(B, H, 1, C)

        x = h * (b.expand_as(h)) + w * (a.expand_as(w)) + (c * (a + b))  # B, H, W, C

        x = x.permute(0, 3, 2, 1)
        # B, C, H, W
        return x



class MDAU(nn.Module):

    def __init__(self,
                 feature_count,
                 in_channels,
                 out_channels,
                 num_block,
                 expansion=4,
                 data_size=64,
                 g1=2,
                 g2=4,
                 norm_cfg=dict(type='BN')):
        super(MDAU, self).__init__()

        self.num_block = num_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion

        self.norm_cfg = norm_cfg
        self.feature_count = feature_count


        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            None,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        # *******************************************************************
        if self.feature_count == 0:  # Layer 1

            seg_dim = g1
            input_size = data_size
            dim = self.mid_channels * input_size // seg_dim
            self.MDA_slow = MDA_MLP(dim=dim, c_dim= self.mid_channels, segment_dim=seg_dim)

            seg_dim = g2
            input_size = data_size
            dim = self.mid_channels * input_size // seg_dim
            self.MDA_fast = MDA_MLP(dim=dim, c_dim= self.mid_channels, segment_dim=seg_dim)

        elif self.feature_count == 1:  # Layer 2

            seg_dim = g1
            input_size = data_size // 2
            dim = self.mid_channels * input_size // seg_dim
            self.MDA_slow = MDA_MLP(dim=dim, c_dim= self.mid_channels, segment_dim=seg_dim)

            seg_dim = g2
            input_size = data_size // 2
            dim = self.mid_channels * input_size // seg_dim
            self.MDA_fast = MDA_MLP(dim=dim, c_dim= self.mid_channels, segment_dim=seg_dim)

        elif self.feature_count == 2:  # Layer 3

            seg_dim = g1
            input_size = data_size // 2
            dim = self.mid_channels * input_size // seg_dim
            self.MDA_slow = MDA_MLP(dim=dim, c_dim= self.mid_channels,segment_dim=seg_dim)

            seg_dim = g2
            input_size = data_size // 2
            dim = self.mid_channels * input_size // seg_dim
            self.MDA_fast = MDA_MLP(dim=dim, c_dim= self.mid_channels, segment_dim=seg_dim)

        else:  # Layer 4

            seg_dim = g1
            input_size = data_size // 4
            dim = self.mid_channels * input_size // seg_dim
            self.MDA_slow = MDA_MLP(dim=dim, c_dim= self.mid_channels, segment_dim=seg_dim)

            seg_dim = g2
            input_size = data_size // 4
            dim = self.mid_channels * input_size // seg_dim
            self.MDA_fast = MDA_MLP(dim=dim, c_dim= self.mid_channels, segment_dim=seg_dim)

        # *******************************************************************
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            None,
            self.mid_channels * 2,
            out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)

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

    def forward(self, x):


        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out1 = self.MDA_fast(out)
        out2 = self.MDA_slow(out)

        out = self.conv3(torch.cat((out1, out2),dim=1))
        out = self.norm3(out)

        if self.num_block >=4:
            out = out + identity

        out = self.relu(out)

        return out


class Layer(nn.Sequential):

    def __init__(self,
                 feature_count,
                 Backbone,
                 num_blocks,
                 in_channels,
                 out_channels,
                 reduction=4,
                 data_size=64,
                 g1=4,
                 g2=8,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        self.block = Backbone
        self.feature_count = feature_count

        layers = []
        for i in range(num_blocks):
            layers.append(
                Backbone(
                    feature_count=self.feature_count,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_block=num_blocks,
                    expansion=reduction,
                    data_size=data_size,
                    g1=g1,
                    g2=g2,
                    norm_cfg=norm_cfg,
                    **kwargs))

        super(Layer, self).__init__(*layers)


@BACKBONES.register_module()
class TModule(BaseBackbone):

    Backbone_maps = {
        'MDAUnit': MDAU
    }

    def __init__(self,
                 layer,
                 layer_config,
                 in_channels=640,
                 reduction=4,
                 data_size=64,
                 g1=4,
                 g2=8,
                 backbone='MDAUnit',
                 out_indices=(3, ),
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 zero_init_residual=True):
        super(TModule, self).__init__()

        assert layer >= 1 and layer <= 4
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.Backbone = self.Backbone_maps[backbone]

        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.stage_blocks = layer_config[:layer]
        self.reduction = reduction

        self._make_stem_layer(in_channels, in_channels)

        self.mda_layers = []

        for i, num_blocks in enumerate(self.stage_blocks):
            if num_blocks == 0:
                mda_layer = nn.Identity()
            else:
                mda_layer = self.make_layer(
                    feature_count=i,
                    Backbone=self.Backbone,
                    num_blocks=num_blocks,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    reduction=self.reduction,
                    data_size=data_size,
                    g1=g1,
                    g2=g2,
                    norm_cfg=norm_cfg)

            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, mda_layer)
            self.mda_layers.append(layer_name)

        self._freeze_stages()


    def make_layer(self, **kwargs):
        return Layer(**kwargs)

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
                conv_cfg=None,
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
                conv_cfg=None,
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

        for i, layer_name in enumerate(self.mda_layers):
            mda_layer = getattr(self, layer_name)
            outs.append(mda_layer(x[i]))
        return outs

    def train(self, mode=True):
        super(TModule, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
