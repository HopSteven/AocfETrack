#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class CSLFF(nn.Module):
    """Channel-Shared Lightweight Feature Fusion (CSLFF)"""
    def __init__(self, channels_list, out_channels=256, act="silu"):
        super().__init__()
        # Proj convs to unify channels to out_channels
        self.proj_conv0 = nn.Conv2d(channels_list[0], out_channels, 1, bias=False)
        self.proj_conv1 = nn.Conv2d(channels_list[1], out_channels, 1, bias=False)
        self.proj_conv2 = nn.Conv2d(channels_list[2], out_channels, 1, bias=False)
        # Fuse conv: 3*out_channels -> 3*out_channels (for per-scale weights)
        self.conv_fuse = nn.Conv2d(out_channels * 3, out_channels * 3, 1, bias=False)
        self.act = nn.SiLU() if act == 'silu' else nn.ReLU()

    def forward(self, feats):  # feats: list of 3 feature maps [pan_out2, pan_out1, pan_out0]
        # Proj to unified channels
        h, w = feats[2].shape[-2:]  # Smallest size
        feat0 = F.interpolate(self.proj_conv0(feats[0]), (h, w), mode='bilinear', align_corners=False)
        feat1 = F.interpolate(self.proj_conv1(feats[1]), (h, w), mode='bilinear', align_corners=False)
        feat2 = self.proj_conv2(feats[2])
        
        # Channel concat
        fused = torch.cat([feat0, feat1, feat2], dim=1)
        weights = self.act(self.conv_fuse(fused))  # [B, 3*C, H, W]
        weights = F.softmax(weights, dim=1)  # Softmax over channels (3*C groups normalized)
        
        # Weighted fusion: slice weights for each feat
        c = feat0.size(1)
        w0, w1, w2 = weights[:, :c], weights[:, c:2*c], weights[:, 2*c:]
        out = w0 * feat0 + w1 * feat1 + w2 * feat2
        return out  # Fused [B, C, H, W]


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = [int(c * width) for c in in_channels]  # [320, 640, 1280] for width=1.25
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            self.in_channels[2], self.in_channels[1], 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * self.in_channels[1]),
            self.in_channels[1],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            self.in_channels[1], self.in_channels[0], 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * self.in_channels[0]),
            self.in_channels[0],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            self.in_channels[0], self.in_channels[0], 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * self.in_channels[0]),
            self.in_channels[1],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            self.in_channels[1], self.in_channels[1], 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * self.in_channels[1]),
            self.in_channels[2],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        
        # CSLFF fusion (pass actual channels)
        self.cslff = CSLFF(self.in_channels, out_channels=self.in_channels[0], act=act)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1280->640/32
        f_out0 = self.upsample(fpn_out0)  # 640/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 640->1280/16
        f_out0 = self.C3_p4(f_out0)  # 1280->640/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 640->320/16
        f_out1 = self.upsample(fpn_out1)  # 320/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 320->640/8
        pan_out2 = self.C3_p3(f_out1)  # 640->320/8

        p_out1 = self.bu_conv2(pan_out2)  # 320->320/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 320->640/16
        pan_out1 = self.C3_n3(p_out1)  # 640->640/16

        p_out0 = self.bu_conv1(pan_out1)  # 640->640/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 640->1280/32
        pan_out0 = self.C3_n4(p_out0)  # 1280->1280/32

        # CSLFF fusion
        feats = [pan_out2, pan_out1, pan_out0]
        fused_feat = self.cslff(feats)
        
        outputs = (fused_feat, pan_out1, pan_out0)  # 返回融合特征 (320) + 原 pan_out1/out0
        return outputs