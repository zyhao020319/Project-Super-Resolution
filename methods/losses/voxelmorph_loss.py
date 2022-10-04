import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCrossCorrelationWithSmoothnessLoss(nn.Module):
    def __init__(self, shape, length, alpha, penalty):
        super(LocalCrossCorrelationWithSmoothnessLoss, self).__init__()
        ndims = len(shape) - 1
        if not ndims in [1, 2, 3]:
            raise AssertionError("volumes should be 1 to 3 dimensions. found: {}".format(ndims))
        win = [length] * ndims
        sum_filt = torch.ones([1, shape[0], *win])
        pad_no = math.floor(win[0] / 2)
        
        self.register_buffer("sum_filt", sum_filt)
        self.stride = [1] * ndims
        self.padding = [pad_no] * ndims
        self.win_size = np.prod(win)
        self.epsilon = 1e-9
        
        print(self.sum_filt)
        print(self.stride)
        print(self.padding)
        print(self.win_size)
        
        self.alpha = alpha
        self.penalty = penalty
        
    def forward(self, I: torch.Tensor, J: torch.Tensor, s: torch.Tensor):
        I_var, J_var, cross = self.compute_local_sums(I, J)
        cc = cross * cross / (I_var * J_var + self.epsilon)
        
        ncc_loss =  -1 * torch.mean(cc)
        smoothness_loss = self.gradient_loss(s) * self.alpha
        total_loss = ncc_loss + smoothness_loss
        
        return total_loss, ncc_loss, smoothness_loss
           
    def compute_local_sums(self, I: torch.Tensor, J: torch.Tensor):
        filt = self.sum_filt
        stride = self.stride
        padding = self.padding
        win_size = self.win_size
        
        I2, J2, IJ = I * I, J * J, I * J
        I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        return I_var, J_var, cross
    
    def gradient_loss(self, s):
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0
 
 
if __name__ == "__main__":  
    lncc = LocalCrossCorrelationWithSmoothnessLoss((1, 128, 128), 9, 0.0, "l2")
    abnormal_points = list()

    x = torch.ones((1, 1, 128, 128), dtype=torch.float32) * (-1)
    y = torch.ones((1, 1, 128, 128), dtype=torch.float32) 
    dvf = torch.rand((1, 2, 128, 128), dtype=torch.float32) * 2 - 1
    loss = lncc.forward(x, y, dvf)
    print(loss[0].item())
