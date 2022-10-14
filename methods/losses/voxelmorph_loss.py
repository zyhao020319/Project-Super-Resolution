import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCrossCorrelationWithSmoothnessLoss(nn.Module):
    """
    定义损失函数，分为相似度损失和平滑度损失两项，
    相似度损失为互相关函数，其互相关范围为周围n^2个元素，
    平滑度损失为相邻像素的差分（空间梯度）
    """
    def __init__(self, shape,   # 输入矩阵形状，不计batch_size维度
                 length,  # 用于计算smoothness loss，平滑滤波器的核大小
                 alpha,  # smoothness loss权重大小
                 penalty  # 用于计算smoothness loss，使用l1 penalty还是l2 penalty
                 ):
        super(LocalCrossCorrelationWithSmoothnessLoss, self).__init__()
        ndims = len(shape) - 1  # 输入维度
        if not ndims in [1, 2, 3]:
            raise AssertionError("volumes should be 1 to 3 dimensions. found: {}".format(ndims))
        win = [length] * ndims  # 形成[length,length]向量
        sum_filt = torch.ones([1, shape[0], *win])  # 形成win维的全1矩阵，此外还有两个维度参数
        pad_no = math.floor(win[0] / 2)  # 取半四舍五入，得到padding
        
        self.register_buffer("sum_filt", sum_filt)
        self.stride = [1] * ndims  # 步长
        self.padding = [pad_no] * ndims  # 填充
        self.win_size = np.prod(win)  # 返回乘积
        self.epsilon = 1e-9  # 防止归零
        
        print(self.sum_filt)
        print(self.stride)
        print(self.padding)
        print(self.win_size)
        
        self.alpha = alpha
        self.penalty = penalty
        
    def forward(self, I: torch.Tensor,  # 预测图层
                J: torch.Tensor,  # 实际图层
                s: torch.Tensor  # 形变场
                ):
        I_var, J_var, cross = self.compute_local_sums(I, J)
        cc = cross * cross / (I_var * J_var + self.epsilon)
        
        ncc_loss =  -1 * torch.mean(cc)
        smoothness_loss = self.gradient_loss(s) * self.alpha
        total_loss = ncc_loss + smoothness_loss
        
        return total_loss, ncc_loss, smoothness_loss

    def compute_local_sums(self, I: torch.Tensor, J: torch.Tensor):
        # 计算相似损失
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
        # 计算平滑损失
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
