import torch
import torch.nn as nn
import re
import types

import torch.nn
import torch.nn.init

from .common import conv1x1_block, Classifier,conv3x3_dw_blockAll,conv3x3_block
from .SE_Attention import *
class FR_PDPpp_block(nn.Module):
    """
    Frequency-Responsive PDP Block (FR-PDP++)
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 1×1 pointwise (pre-projection)
        self.Pw1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=in_channels,
            use_bn=False,
            activation=None
        )

        # multi-scale depthwise convolutions
        self.Dw3 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.Dw5 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=5, stride=stride, padding=2,
            groups=in_channels, bias=False
        )
        self.Dw7 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=7, stride=stride, padding=3,
            groups=in_channels, bias=False
        )

        # fusion 1×1
        self.Pw2 = conv1x1_block(
            in_channels=3 * in_channels,
            out_channels=out_channels,
            groups=1
        )

        # SE attention
        self.SE = SE(out_channels, 16)

        # residual projection (if needed)
        self.PwR = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride
        )

    def forward(self, x):
        residual = x

        # pointwise
        x = self.Pw1(x)

        # multi-scale depthwise
        x3 = self.Dw3(x)
        x5 = self.Dw5(x)
        x7 = self.Dw7(x)

        # concat along channel dim
        x = torch.cat([x3, x5, x7], dim=1)

        # fusion
        x = self.Pw2(x)

        # attention
        x = self.SE(x)

        # residual
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else:
            x = x + self.PwR(residual)

        return x

class TickNet(torch.nn.Module):
    """
    Class for constructing TickNet.    
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1                
                stage.add_module("unit{}".format(unit_id + 1), FR_PDPpp_block(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        self.final_conv_channels = 1024        
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        in_channels = self.final_conv_channels
        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

###
#%% model definitions
###
def build_TickNet(num_classes, typesize='small', cifar=False):
    init_conv_channels = 32
    if typesize=='basic':
        channels = [[128],[64],[128],[256],[512]]
    if typesize=='small':
        channels = [[128],[64,128],[256,512,128],[64,128,256],[512]]
    if typesize=='large':
        channels = [[128],[64,128],[256,512,128,64,128,256],[512,128,64,128,256],[512]]
    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        if typesize=='basic':
            strides = [1, 2, 2, 2, 2]
        else:
            strides = [2, 1, 2, 2, 2]
    return  TickNet(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides,
                       in_size=in_size)
