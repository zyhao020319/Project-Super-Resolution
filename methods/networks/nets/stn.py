import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformationNetwork(nn.Module):
    def __init__(self, size):
        super(SpatialTransformationNetwork, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]

        output = F.grid_sample(src, new_locs, mode="bilinear", align_corners=True)
        return output
