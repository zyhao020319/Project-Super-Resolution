"""
    @Author : Yuan Dalong
    @Description : 实现随机选取若干个模态的slice或volume
"""

import random

import numpy as np
import torch


def slice_collater(batch):  # 传入函数为一批次的数据集
    batch_moving_images = list()
    batch_fixed_images = list()
    batch_are_selected = list()
    for seq_slice_0, seq_slice_1 in batch:  # seq_slice_0为初始切面，seq_slice_1为结束切面
        is_selected = [True, True, True, True]
        exist_pos, nonexist_pos = random.sample([0, 1, 2, 3], k=2)  # 从四个数中随机抽取两个数
        is_selected[exist_pos] = True
        is_selected[nonexist_pos] = False
        for i in range(3):
            if i in [exist_pos, nonexist_pos]:
                continue
            else:  # 50%概率将本位置1或0
                if random.random() < 0.5:
                    is_selected[i] = True
                else:
                    is_selected[i] = False

        sequence_0 = [seq_slice_0[i: i + 1] for i in range(len(seq_slice_0))]  # 将tensor的第0个维度切换为list
        sequence_1 = [seq_slice_1[i: i + 1] for i in range(len(seq_slice_1))]

        moving_image, fixed_image = list(), list()
        for i, _is_selected in enumerate(is_selected):  # 构建序列，返回元素的索引和元素本身
            if _is_selected:
                moving_image.append(sequence_0[i])
                fixed_image.append(sequence_1[i])
            else:
                moving_image.append(torch.zeros_like(sequence_0[i]))
                fixed_image.append(torch.zeros_like(sequence_1[i]))
        moving_image = torch.cat(moving_image)
        fixed_image = torch.cat(fixed_image)
        is_selected = torch.from_numpy(np.array(is_selected))

        batch_moving_images.append(moving_image)
        batch_fixed_images.append(fixed_image)
        batch_are_selected.append(is_selected)

    return torch.stack(batch_moving_images), torch.stack(batch_fixed_images), torch.stack(batch_are_selected)


def volume_collater(batch):
    batch_moving_images = list()
    batch_fixed_images = list()
    batch_are_selected = list()
    for seq_volume_0, seq_volume_1 in batch:
        is_selected = [True, True, True, True]
        exist_pos, nonexist_pos = random.sample([0, 1, 2, 3], k=2)
        is_selected[exist_pos] = True
        is_selected[nonexist_pos] = False
        for i in range(3):
            if i in [exist_pos, nonexist_pos]:
                continue
            else:
                if random.random() < 0.5:
                    is_selected[i] = True
                else:
                    is_selected[i] = False

        sequence_0 = [seq_volume_0[i: i + 1] for i in range(len(seq_volume_0))]
        sequence_1 = [seq_volume_1[i: i + 1] for i in range(len(seq_volume_1))]

        moving_image, fixed_image = list(), list()
        for i, _is_selected in enumerate(is_selected):
            if _is_selected:
                moving_image.append(sequence_0[i])
                fixed_image.append(sequence_1[i])
            else:
                moving_image.append(torch.zeros_like(sequence_0[i]))
                fixed_image.append(torch.zeros_like(sequence_1[i]))
        moving_image = torch.cat(moving_image)
        fixed_image = torch.cat(fixed_image)
        is_selected = torch.from_numpy(np.array(is_selected))

        batch_moving_images.append(moving_image)
        batch_fixed_images.append(fixed_image)
        batch_are_selected.append(is_selected)

    return torch.stack(batch_moving_images), torch.stack(batch_fixed_images), torch.stack(batch_are_selected)
