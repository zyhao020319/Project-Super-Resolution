<<<<<<< HEAD
import os
import enum
=======
"""
    @Author : Yuan Dalong
    @Description : 实现BraTs2018数据集的4个模态的切片读取类，继承自torch.utils.data.Dataset，其__getitem__函数返回大小为[4, 240, 240]的torch.Tensor张量
"""

import os
>>>>>>> origin/main

import numpy as np
import torch
import torchio as tio
<<<<<<< HEAD

from experiments.dataset_configs.brats2018slice_configs import MODALITY, STAGE


# class MODALITY(enum.Enum):
#     T1 = "t1"
#     T1CE = "t1ce"
#     T2 = "t2"
#     FLAIR = "flair"
#
#
# class STAGE(enum.Enum):
#     TRAIN = "train"
#     VAL = "val"
#     TEST = "test"


class BraTs2018DatasetSlice(tio.SubjectsDataset):
    def __init__(self, sr_scale: int, stage: STAGE):
        self.data_dir = "../datasets/BraTs2018"  # 改为本项目的上级目录下的datasets文件夹

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
        super().__init__(subjects)
=======
from torch.utils.data import Dataset


class BraTs2018Slice(Dataset):
    def __init__(self, stages: list, sr_scale: int = 8):
        """BraTs2018多模态切片读取类
        Args:
            stages (list): 训练的阶段，设置成list是为了兼容GAN
            sr_scale (int, optional): 超分的倍数. Defaults to 8.
        """
        super().__init__()
        self.data_dir = "../datasets/BraTs2018"  # 改为本项目的上级目录下的datasets文件夹

        names_and_grades = self.get_cases(stages=stages)  # 以list形式返回特定stage下的数据name和label
        self.modals = ["t1", "t1ce", "t2", "flair"]  # 定义四个模态
        _multimodal_paths = [
            os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{grade_}", "{name_}",
                                        "{name_}" + "_" + modal + ".nii.gz")
            # 使用名称name_，grade_格式化
            for modal in self.modals]  # 初始化多模态数据路径

        subjects = self.get_subjects(names_and_grades=names_and_grades, _multimodal_paths=_multimodal_paths)
        # 根据传递的name和label在指定路径下实例化数据集
        self.subjects = tio.SubjectsDataset(subjects=subjects)  # 将数据集打包成dataset
>>>>>>> origin/main

        self.image_shape = (240, 240, 155)  # 实例化3D图像大小
        self.sr_scale = sr_scale  # 超分倍数
        self.num_used_slice_pairs = self.image_shape[-1] // sr_scale  # 实例化用于切片的层数
        self.start_slice = (self.image_shape[-1] - sr_scale * self.num_used_slice_pairs - 1) // 2  # 实例化起始层

<<<<<<< HEAD
    def __getitem__(self, idx: int):
        # 迭代器返回对应subject的切片以及下一切片
        index, forward_or_backward = divmod(idx, 2)
        subject_idx, slice_idx = divmod(index, self.num_used_slice_pairs)
        subject = super().__getitem__(index=subject_idx)

        slice_pair = self.get_slice_pair(subject, slice_idx, forward_or_backward)
        return slice_pair

    def get_slice_pair(self, subject: tio.Subject, slice_idx: int, forward_or_backward: int):
        # 前项或后项返回对应subject的切片以及下一切片
        t1 = subject["t1"].data
        t1ce = subject["t1ce"].data
        t2 = subject["t2"].data
        flair = subject["flair"].data

        image = torch.cat([t1, t1ce, t2, flair])
        image = image.float() / 32767 * 2 - 1

        slice_0 = image[..., self.start_slice + self.sr_scale * slice_idx]
        slice_1 = image[..., self.start_slice + self.sr_scale * (slice_idx + 1)]
        if forward_or_backward == 0:
            return slice_0, slice_1
        else:
            return slice_1, slice_0

    def __len__(self):
        return self.num_used_slice_pairs * super().__len__() * 2

    def get_subjects(self):
        # 读取所选阶段的多模态MR图像到内存中
        subjects = list()
        for case_name in self.case_names:
            t1_path = self._t1_path.format(grade_=case_name[1], name_=case_name[0],)
            t1ce_path = self._t1ce_path.format(grade_=case_name[1], name_=case_name[0])
            t2_path = self._t2_path.format(grade_=case_name[1], name_=case_name[0])
            flair_path = self._flair_path.format(grade_=case_name[1], name_=case_name[0])
=======
    def __getitem__(self, index: int):
        subject_idx, slice_idx = divmod(index, self.num_used_slice_pairs)
        # ----------------------------------------------------------------#
        # subject_idx : 被选中的subject的序号
        # slice_idx   : 被选中的切片对的序号
        # ----------------------------------------------------------------#

        subject = self.subjects[subject_idx]

        slice_0 = list()  # 初始切面
        slice_1 = list()  # 结束切面
        for modal in self.modals:
            image = subject[modal].data

            slice_0.append(image[..., self.start_slice + self.sr_scale * slice_idx])
            slice_1.append(image[..., self.start_slice + self.sr_scale * (slice_idx + 1)])
        slice_0 = torch.cat(slice_0).float() / 32767 * 2 - 1  # 将16bit图像归一化至-1到1中
        slice_1 = torch.cat(slice_1).float() / 32767 * 2 - 1

        return slice_0, slice_1

    def __len__(self):
        return self.num_used_slice_pairs * len(self.subjects)

    @staticmethod
    def get_subjects(names_and_grades: list, _multimodal_paths: list):
        """读取所选阶段的多模态MR图像到内存中

        Args:
            names_and_grades (list): [[name0, grade0], [name1, grade1], ...]
            _multimodal_paths (list): [_t1_path_format, _t1ce_path_format, _t2_path_format, _flair_path_format]

        Returns:
            list[tio.Subject]: tio.Subject的list，每个Subject包含t1图像、t1ce图像、t2图像和flair图像
        """
        subjects = list()
        for name, grade in names_and_grades:
            '''
            [['Brats18_TCIA04_437_1', 'HGG'],...] 
            '''
            t1_path, t1ce_path, t2_path, flair_path = [
                _modal_path.format(grade_=grade, name_=name) for _modal_path in _multimodal_paths
                # 使用名称name_，grade_格式化
            ]  # 实例化所选stage下的每个文件路径
>>>>>>> origin/main

            subject = tio.Subject(
                t1=tio.ScalarImage(t1_path),
                t1ce=tio.ScalarImage(t1ce_path),
                t2=tio.ScalarImage(t2_path),
                flair=tio.ScalarImage(flair_path),
<<<<<<< HEAD
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
=======
                grade=grade,
                name=name
            )  # 使用torchio的Subject类读取MR图像
            subjects.append(subject)
        return subjects

    def get_cases(self, stages: list):  # stage是数据集种类，分为train、val、test
        # 用于读取指定stage下的数据集索引
        case_names = list()
        for stage in stages:
            stage_case_names = self.parse_txt(os.path.join(self.data_dir, "ImageSets/Main/" + stage + ".txt"))
            case_names += stage_case_names
        return case_names

    @staticmethod
    def parse_txt(txt_path: str):  # 从数据集索引txt文件中读取特定数据的name和label
>>>>>>> origin/main
        lines = np.loadtxt(txt_path, dtype=np.str_)
        rets = list()
        for line in lines:
            t1_path = line[0]
            case_name = os.path.split(os.path.split(t1_path)[0])[-1]
            grade = line[-1]
            rets.append([case_name, grade])
        return rets
