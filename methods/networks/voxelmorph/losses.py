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
        self.losses += [Grad(penalty, loss_mult=int_downsize)]
        self.weights += [weight]
        
    def forward(self, pred_true_pairs_with_dvf: list):
        losses = list()
        for elem, loss_func, weight in zip(pred_true_pairs_with_dvf, self.losses, self.weights):
            losses.append(weight * loss_func(*elem))
        return sum(losses), [loss.item() for loss in losses]
        

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

    def forward(self, y_true, y_pred):
        loss = 0.0

        for modality_num in range(y_true.shape[1]):
            Ii = y_true[:, modality_num : modality_num + 1, ...]
            Ji = y_pred[:, modality_num : modality_num + 1, ...]

            # compute CC squares
            I2 = Ii * Ii
            J2 = Ji * Ji
            IJ = Ii * Ji

            I_sum = self.conv_fn(Ii, self.sum_filt, stride=self.stride, padding=self.padding)
            J_sum = self.conv_fn(Ji, self.sum_filt, stride=self.stride, padding=self.padding)
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

        return loss / modality_num


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

