import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformationNetwork(nn.Module):
    '''
    实现将unet学习出的特征矩阵作用于原图坐标系下
    '''
    def __init__(self, size):
        super(SpatialTransformationNetwork, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)  # 生成两个0-223坐标点矩阵
        grid = torch.stack(grids)  # 拼接这两个矩阵
        grid = torch.unsqueeze(grid, 0)  # 在dim=0维度升维变成四维
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)
        '''
        self.register_buffer可以将tensor注册成buffer,在forward中使用self.mybuffer,而不是self.mybuffer_tmp
        网络存储时也会将buffer存下，当网络load模型时，会将存储的模型的buffer也进行赋值。
        buffer的更新在forward中，optim.step只能更新nn.parameter类型的参数。
        '''

    def forward(self, src, flow):
        new_locs = self.grid + flow  # flow为Unet学习出的特征图片
        shape = flow.shape[2:]  # 读取图片大小

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)  # 对dim=1归一到[-1,1]区间
        new_locs = new_locs.permute(0, 2, 3, 1)  # 重新排列dim=1
        new_locs = new_locs[..., [1, 0]]

        output = F.grid_sample(src, new_locs, mode="bilinear", align_corners=True)
        '''
        提供一个input的Tensor以及一个对应的flow-field网格(比如光流，体素流等)，
        然后根据grid中每个位置提供的坐标信息(这里指input中pixel的坐标)，
        将input中对应位置的像素值填充到grid指定的位置，得到最终的输出.
        
        对于mode='bilinear'参数，则定义了在input中指定位置的pixel value中进行双线性插值的方法
        
        默认padding_mode为zero填充
        '''
        return output
