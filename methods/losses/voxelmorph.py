"""
    @Author : Yuan Dalong
    @Description : 实现任意模态的MR图像的损失函数计算
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn


class NCCWithGrad(nn.Module):
    def __init__(self, bidir, int_downsize, weight, win=None, penalty="l2"):
        super(NCCWithGrad, self).__init__()

        self.losses = list()
        self.weights = list()        
        self.image_loss_func = NCC(win)

        # need two image loss functions if bidirectional
        if bidir:
            self.losses += [self.image_loss_func, self.image_loss_func]
            self.weights += [0.5, 0.5]
        else:
            self.losses += [self.image_loss_func]
            self.weights = [1]

        # prepare deformation loss
        self.losses += [Grad(penalty, loss_mult=int_downsize), Grad(penalty, loss_mult=int_downsize)]
        self.weights += [0.5 * weight, 0.5 * weight]
        
    def forward(self, pred_true_pairs_with_dvf: list):
        """
        Args:
            pred_true_pairs_with_dvf (list): 
            [
                (batch_warped_moving_images, batch_fixed_images, batch_are_selected), 
                (batch_warped_fixed_images, batch_moving_images, batch_are_selected), 
                (forward_dvf,)
                (backward_dvf,)
            ]
        Returns:
            tuple: loss以及loss的各个子项
        """
        losses = list()
        for elem, loss_func, weight in zip(pred_true_pairs_with_dvf, self.losses, self.weights):
            losses.append(weight * loss_func(*elem))
        
        return sum(losses), [sum(loss.item() for loss in losses[:2]), sum(loss.item() for loss in losses[2:])]
        

class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, win=None):  
        super(NCC, self).__init__()
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = 2

        # set window size
        self.win = [9] * ndims if win is None else win

        # compute filters
        self.register_buffer("sum_filt", torch.ones([1, 1, *self.win]))
        pad_no = math.floor(self.win[0] / 2)
        self.stride = (1, 1)
        self.padding = (pad_no, pad_no)

        # get convolution function
        self.conv_fn = F.conv2d

    def forward(self, y_true : torch.Tensor, y_pred : torch.Tensor, are_selected : torch.Tensor):
        loss = 0.0

        for _y_true, _y_pred, _are_selected in zip(y_true, y_pred, are_selected):           
            Ii = _y_true[_are_selected].unsqueeze(1)
            Ji = _y_pred[_are_selected].unsqueeze(1)

            # compute CC squares
            I2 = Ii * Ii
            J2 = Ji * Ji
            IJ = Ii * Ji

            I_sum  = self.conv_fn(Ii, self.sum_filt, stride=self.stride, padding=self.padding)
            J_sum  = self.conv_fn(Ji, self.sum_filt, stride=self.stride, padding=self.padding)
            I2_sum = self.conv_fn(I2, self.sum_filt, stride=self.stride, padding=self.padding)
            J2_sum = self.conv_fn(J2, self.sum_filt, stride=self.stride, padding=self.padding)
            IJ_sum = self.conv_fn(IJ, self.sum_filt, stride=self.stride, padding=self.padding)

            win_size = np.prod(self.win)
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            cc = cross * cross / (I_var * J_var + 1e-5)
            loss -= torch.mean(cc)

        return loss / y_true.shape[0]


class MSE:
    """
    Mean squared error loss.
    """
    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """
    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad(nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, dvf: torch.Tensor):
        dy = torch.abs(dvf[:, :, 1:, :] - dvf[:, :, :-1, :])
        dx = torch.abs(dvf[:, :, :, 1:] - dvf[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = (torch.mean(dx) + torch.mean(dy)) / 2
        if self.loss_mult is not None:
            d *= self.loss_mult
        return d       

