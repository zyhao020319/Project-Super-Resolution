"""
    @Author : Yuan Dalong
    @Description : 实现BraTs2018数据集的4个模态的读取类，继承自torch.utils.data.Dataset，其__getitem__函数返回大小为[4, 240, 240, 155]的torch.Tensor张量
"""

import os

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset


class BraTs2018Volume(Dataset):
    def __init__(self, stages : list, sr_scale : int = 8):
        """BraTs2018多模态低分辨率三维图像读取类，主要是为了测试的时候用
        Args:
            stages (list): 训练的阶段，设置成list是为了兼容GAN
            sr_scale (int, optional): 超分的倍数. Defaults to 8.
        """
        super().__init__()
        self.data_dir    = "/dat01/zhangjunyu/work/ydl/GliomaSegmentationAndGrading/datasets/BraTs2018"  
              
        names_and_grades  = self.get_cases(stages=stages)
        self.modals       = ["t1", "t1ce", "t2", "flair"]
        _multimodal_paths = [os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{:s}", "{:s}", "{:s}_" + modal + ".nii.gz") for modal in self.modals]

        subjects      = self.get_subjects(names_and_grades=names_and_grades, _multimodal_paths=_multimodal_paths)
        self.subjects = tio.SubjectsDataset(subjects=subjects)
        
        self.image_shape          = (240, 240, 155)
        self.sr_scale             = sr_scale
        self.num_used_slice_pairs = self.image_shape[-1] // sr_scale
        self.start_slice          = (self.image_shape[-1] - sr_scale * self.num_used_slice_pairs - 1) // 2

    def __getitem__(self, index: int):
        subject = self.subjects[index]
        
        t1 = subject["t1"].data
        t1ce = subject["t1ce"].data
        t2 = subject["t2"].data
        flair = subject["flair"].data
        
        image = torch.cat([t1, t1ce, t2, flair])
        image = image.float() / 32767 * 2 - 1        
        
        slice_idxs = list(range(self.start_slice, self.image_shape[-1], self.sr_scale))
        lr_image = image[..., slice_idxs]
        hr_image = image[..., self.start_slice:slice_idxs[-1] + 1]
        return lr_image, hr_image

    def __len__(self):
        return len(self.subjects)
        
    def get_subjects(self, names_and_grades : list, _multimodal_paths : list):
        """读取所选阶段的多模态MR图像到内存中
        
        Args:
            names_and_grades (list): [[name0, grade0], [name1, grade1], ...]
            _multimodal_paths (list): [_t1_path_format, _t1ce_path_format, _t2_path_format, _flair_path_format]

        Returns:
           list[tio.Subject]: tio.Subject的list，每个Subject包含t1图像、t1ce图像、t2图像和flair图像
        """
        subjects = list()
        for name, grade in names_and_grades:
            t1_path, t1ce_path, t2_path, flair_path = [
                _modal_path.format(grade, name, name) for _modal_path in _multimodal_paths
            ]
            
            subject = tio.Subject(
                t1=tio.ScalarImage(t1_path),
                t1ce=tio.ScalarImage(t1ce_path),
                t2=tio.ScalarImage(t2_path),
                flair=tio.ScalarImage(flair_path),
                grade=grade,
                name=name
            )
            subjects.append(subject) 
        return subjects

    def get_cases(self, stages : list):  
        case_names = list()
        for stage in stages:
            stage_case_names = self.parse_txt(os.path.join(self.data_dir, "ImageSets/Main/" + stage +".txt"))
            case_names += stage_case_names
        return case_names
    
    @staticmethod
    def parse_txt(txt_path: str):
        lines = np.loadtxt(txt_path, dtype=np.str_)
        rets = list()
        for line in lines:
            t1_path = line[0]
            case_name = os.path.split(os.path.split(t1_path)[0])[-1]
            grade = line[-1]
            rets.append([case_name, grade])
        return rets           