from pkgutil import ImpImporter
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.networks.nets.unet import UNet
from methods.networks.nets.stn import SpatialTransformationNetwork


class SliceInterpolationNet(nn.Module):
    def __init__(self, in_channels, out_channels, ngf, input_shape):
        super(SliceInterpolationNet, self).__init__()
        
        self.unet = UNet(in_channels, out_channels, (3, 3), ngf, nlayers=4, bilinear=True)
        self.stn  = SpatialTransformationNetwork(size=input_shape)
        
    def forward(self, moving_image: torch.Tensor, fixed_image: torch.Tensor):
        #-----------------------------------------#
        # x为moving image和fixed image的堆叠
        # x : [batch_size, 2, 240, 240]
        #-----------------------------------------#
        x = torch.cat([moving_image, fixed_image], dim=1)
        dvf = self.unet.forward(x)
        warped_image = self.stn.forward(moving_image, dvf)
        return warped_image, dvf
    
    def slice_interplolation(self, prev_slice: torch.Tensor, next_slice: torch.Tensor, t: float):
        assert 0 < t < 1
        _, forward_dvf = self.forward(moving_image=prev_slice, fixed_image=next_slice)
        _, backward_dvf = self.forward(moving_image=next_slice, fixed_image=prev_slice)
        dvf_0_to_t = forward_dvf * t
        dvf_1_to_t = backward_dvf * (1 - t)
        return self.stn(prev_slice, dvf_0_to_t) + self.stn(next_slice, dvf_1_to_t)
        