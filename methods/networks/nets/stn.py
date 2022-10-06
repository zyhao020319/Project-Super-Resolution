import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformationNetwork(nn.Module):
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
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]

        output = F.grid_sample(src, new_locs, mode="bilinear", align_corners=True)
        return output
