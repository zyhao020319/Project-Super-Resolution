import os

import torch
import numpy as np
from torch.utils.data import Dataset
import torchio as tio

from experiments.dataset_configs.brats2018slice_configs import STAGE


# class STAGE(enum.Enum):
#     TRAIN = "train"
#     VAL = "val"
#     TEST = "test"


class BraTs2018Volume(Dataset):
    def __init__(self, sr_scale: int, stage: STAGE):
        super().__init__()
        self.data_dir = "../datasets/BraTs2018"

        self.stage = stage
        self.case_names = self.get_cases()

        self._t1_path = os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{grade_}", "{name_}",
                                     "{name_}_t1.nii.gz")
        self._t1ce_path = os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{grade_}", "{name_}",
                                       "{name_}_t1ce.nii.gz")
        self._t2_path = os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{grade_}", "{name_}",
                                     "{name_}_t2.nii.gz")
        self._flair_path = os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{grade_}", "{name_}",
                                        "{name_}_flair.nii.gz")
        # 使用名称name_，grade_格式化

        subjects = self.get_subjects()
        # 根据传递的self.case_names在指定路径下实例化数据集
        self.subjects = tio.SubjectsDataset(subjects)

        self.image_shape = (240, 240, 155)  # 实例化3D图像大小
        self.sr_scale = sr_scale  # 超分倍数
        self.num_used_slice_pairs = self.image_shape[-1] // sr_scale  # 实例化用于切片的层数
        self.start_slice = (self.image_shape[-1] - sr_scale * self.num_used_slice_pairs - 1) // 2  # 实例化起始层

    def __getitem__(self, index: int):
        # 迭代器返回对应低分辨率图像和高分辨率图像
        subject = self.subjects[index]

        t1 = subject["t1"].data
        t1ce = subject["t1ce"].data
        t2 = subject["t2"].data
        flair = subject["flair"].data

        image = torch.cat([t1, t1ce, t2, flair])
        image = image.float() / 32767 * 2 - 1

        slice_idxs = list(range(self.start_slice, self.image_shape[-1], self.sr_scale))
        lr_image = image[..., slice_idxs]
        hr_image = image[..., self.start_slice:slice_idxs[-1]]
        return lr_image, hr_image

    def __len__(self):
        return len(self.subjects)

    def get_subjects(self):
        # 读取所选阶段的多模态MR图像到内存中
        subjects = list()
        for case_name in self.case_names:
            t1_path = self._t1_path.format(grade_=case_name[1], name_=case_name[0],)
            t1ce_path = self._t1ce_path.format(grade_=case_name[1], name_=case_name[0])
            t2_path = self._t2_path.format(grade_=case_name[1], name_=case_name[0])
            flair_path = self._flair_path.format(grade_=case_name[1], name_=case_name[0])

            subject = tio.Subject(
                t1=tio.ScalarImage(t1_path),
                t1ce=tio.ScalarImage(t1ce_path),
                t2=tio.ScalarImage(t2_path),
                flair=tio.ScalarImage(flair_path),
                grade=case_name[1],
                name=case_name[0]
            )
            subjects.append(subject)
        return subjects

    def get_cases(self):
        # 用于读取指定stage下的数据集索引
        if isinstance(self.stage, list):
            case_names = list()
            for stage in self.stage:
                stage: STAGE
                stage_case_names = self.parse_txt(os.path.join(self.data_dir, "ImageSets/Main/" + stage.value + ".txt"))
                case_names += stage_case_names
        else:
            case_names = self.parse_txt(os.path.join(self.data_dir, "ImageSets/Main/" + self.stage.value + ".txt"))
        return case_names

    @staticmethod
    def parse_txt(txt_path: str):
        # 从数据集索引txt文件中读取特定数据的name和label
        lines = np.loadtxt(txt_path, dtype=np.str_)
        rets = list()
        for line in lines:
            t1_path = line[0]
            case_name = os.path.split(os.path.split(t1_path)[0])[-1]
            grade = line[-1]
            rets.append([case_name, grade])
        return rets
