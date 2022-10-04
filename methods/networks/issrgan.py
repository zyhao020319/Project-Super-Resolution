import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution

from methods.networks.nets.unet2d5 import UNet2d5
from methods.networks.nets.resnet import resnet18


class ISSRGenerator(nn.Module):
    def __init__(self, in_channels, sr_scale, 
                 num_res_units=2,
                 channels=(16, 32, 48, 64, 80, 96),
                 strides=((2, 2, 1), (2, 2, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2),),
                 kernel_sizes=((3, 3, 1), (3, 3, 1), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3),),
                 sample_kernel_sizes=((3, 3, 1), (3, 3, 1), (3, 3, 3), (3, 3, 3), (3, 3, 3),)
                 ):
        super(ISSRGenerator, self).__init__()
        self.in_channels = in_channels
        self.sr_scale = sr_scale
        
        out_channels = in_channels * sr_scale
        self.backbone = UNet2d5(dimensions=3, 
                                in_channels=in_channels,
                                out_channels=out_channels,
                                channels=channels,
                                strides=strides,
                                kernel_sizes=kernel_sizes,
                                sample_kernel_sizes=sample_kernel_sizes,
                                num_res_units=num_res_units)
        self.head = nn.Tanh()
        
    def forward(self, x: torch.Tensor):
        y = self.head(self.backbone(x))
        
        y: torch.Tensor
        batch_size, out_channels, height, width, depth = y.size()
        y = y.reshape(batch_size, self.in_channels, self.sr_scale, height, width, depth)
        y = y.permute(0, 1, 3, 4, 5, 2)
        sr_depth = self.sr_scale * depth
        y = y.reshape(batch_size, self.in_channels, height, width, sr_depth)
        
        return y
    
    
class ISSRDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(ISSRDiscriminator, self).__init__()
        self.backbone = resnet18(in_channels=in_channels)
        self.head = Convolution(spatial_dims=2, in_channels=256, out_channels=1, strides=1, kernel_size=1, norm=None, act=None)
        
    def forward(self, x: torch.Tensor):
        y = self.head(self.backbone(x))
        return y
        