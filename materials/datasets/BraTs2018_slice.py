import os
import json
import functools
import enum
import random

import numpy as np
import torch
import torchio as tio


class MODALITY(enum.Enum):
    T1 = "t1"
    T1CE = "t1ce"
    T2 = "t2"
    FLAIR = "flair"


class STAGE(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"



class BraTs2018DatasetSlice(tio.SubjectsDataset):
    def __init__(self, sr_scale: int, stage: STAGE):
        self.data_dir    = "/dat01/zhangjunyu/work/ydl/GliomaSegmentationAndGrading/datasets/BraTs2018"
        
        self.stage                = stage
        self.case_names           = self.get_cases()

        self._t1_path             = os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{:s}", "{:s}", "{:s}_t1.nii.gz")
        self._t1ce_path           = os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{:s}", "{:s}", "{:s}_t1ce.nii.gz")
        self._t2_path             = os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{:s}", "{:s}", "{:s}_t2.nii.gz")
        self._flair_path          = os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{:s}", "{:s}", "{:s}_flair.nii.gz")

        subjects = self.get_subjects()                
        super().__init__(subjects)
        
        self.image_shape          = (240, 240, 155)
        self.sr_scale             = sr_scale
        self.num_used_slice_pairs = self.image_shape[-1] // sr_scale
        self.start_slice          = (self.image_shape[-1] - sr_scale * self.num_used_slice_pairs - 1) // 2
        
    def __getitem__(self, idx: int):
        index, forward_or_backward = divmod(idx, 2)
        subject_idx, slice_idx = divmod(index, self.num_used_slice_pairs)
        subject = super().__getitem__(index=subject_idx)
        
        slice_pair = self.get_slice_pair(subject, slice_idx, forward_or_backward)
        return slice_pair
    
    def get_slice_pair(self, subject: tio.Subject, slice_idx: int, forward_or_backward: int):
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
        subjects = list()
        for case_name in self.case_names:
            t1_path = self._t1_path.format(case_name[1], case_name[0], case_name[0])
            t1ce_path = self._t1ce_path.format(case_name[1], case_name[0], case_name[0])
            t2_path = self._t2_path.format(case_name[1], case_name[0], case_name[0])
            flair_path = self._flair_path.format(case_name[1], case_name[0], case_name[0])
            
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
        if isinstance(self.stage, list):
            case_names = list()
            for stage in self.stage:
                stage : STAGE
                stage_case_names = self.parse_txt(os.path.join(self.data_dir, "ImageSets/Main/" + stage.value +".txt"))
                case_names += stage_case_names
        else:
            case_names = self.parse_txt(os.path.join(self.data_dir, "ImageSets/Main/" + self.stage.value +".txt"))
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
