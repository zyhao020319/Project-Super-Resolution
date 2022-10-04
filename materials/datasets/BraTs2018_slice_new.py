import os
import json
import functools
import enum
import random

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset


class MODALITY(enum.Enum):
    T1 = "t1"
    T1CE = "t1ce"
    T2 = "t2"
    FLAIR = "flair"


class STAGE(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"



class BraTs2018DatasetSliceNew(Dataset):
    def __init__(self, sr_scale: int, modality: MODALITY, stage: STAGE):
        self.data_dir    = "materials/datasets/BraTs2018"
        
        self.image_shape          = (240, 240, 155)
        self.sr_scale             = sr_scale
        self.modality             = modality
        self.stage                = stage
        self._imgpath             = os.path.join(self.data_dir, "MICCAI_BraTS_2018_Data_Training", "{:s}", "{:s}", "{:s}_{:s}.nii.gz")
        
        self.case_names = self.get_cases()
        
        self.slice_pairs = list()
        subjects = tio.SubjectsDataset(self.get_subjects())
        self.get_slice_pairs(subjects)
        
    def get_slice_pairs(self, subjects: tio.SubjectsDataset):
        for subject in subjects:
            subject : tio.Subject
            
            img = subject["image"].data
            img : torch.Tensor
            img = img.float() / 32767 * 2 - 1

            num_used_slice_pairs = img.shape[-1] // self.sr_scale
            start_slice          = (img.shape[-1] - self.sr_scale * num_used_slice_pairs - 1) // 2            
            
            for slice_idx in range(num_used_slice_pairs):
                slice_0 = img[..., start_slice + self.sr_scale * slice_idx]
                slice_1 = img[..., start_slice + self.sr_scale * (slice_idx + 1)]
                
                self.slice_pairs.append((slice_0, slice_1))
                
    def __getitem__(self, idx: int):
        index, forward_or_backward = divmod(idx, 2)
        
        slice_pair = self.get_slice_pair(index, forward_or_backward)
        return slice_pair
    
    def get_slice_pair(self, index: int, forward_or_backward: int):
        slice_0, slice_1 = self.slice_pairs[index]
        if forward_or_backward == 0:
            return slice_0, slice_1
        else:
            return slice_1, slice_0
         
    def __len__(self):
        return len(self.slice_pairs) * 2

    def get_subjects(self):
        subjects = list()
        for case_name in self.case_names:
            imgpath = self._imgpath.format(case_name[1], case_name[0], case_name[0], self.modality.value)
            
            subject = tio.Subject(
                image=tio.ScalarImage(imgpath),
                name=case_name
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
