import os
import enum
from matplotlib import image

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

class Preprocessing:
    def __init__(self, min_nonzero_pixels: int):
        self.min_nonzero_pixels = min_nonzero_pixels
        
        self.src_data_dir = "/dat01/zhangjunyu/work/ydl/GliomaSegmentationAndGrading/datasets/BraTs2018/"
        self.dst_data_dir = "materials/datasets/BraTs2018"
        
        self.image_shape = (240, 240, 155)
        self.stage = [STAGE.TRAIN, STAGE.VAL, STAGE.TEST]
        self.modalities = ["t1", "t1ce", "t2", "flair"]
        self._src_paths = [os.path.join(self.src_data_dir, "MICCAI_BraTS_2018_Data_Training", "{:s}", "{:s}", "{:s}_" + modality + ".nii.gz") for modality in self.modalities]
        self._dst_paths = [os.path.join(self.dst_data_dir, "MICCAI_BraTS_2018_Data_Training", "{:s}", "{:s}", "{:s}_" + modality + ".nii.gz") for modality in self.modalities]

        self.case_names = self.get_cases()
        
        subjects = self.get_subjects()             
        self.subjects = tio.SubjectsDataset(subjects)
    
    def filter_slices(self, img: torch.Tensor):
        _, _, depth = img.shape[-3:]
        for i in range(depth):
            slice = img[..., i]
            if torch.count_nonzero(slice) > self.min_nonzero_pixels:
                start = i
                break
        
        for i in range(depth - 1, -1, -1):
            slice = img[..., i]
            if torch.count_nonzero(slice) > self.min_nonzero_pixels:
                end = i
                break
            
        return start, end
    
    def save_preprocessed_images(self):
        for subject in self.subjects:
            grade = subject["grade"]
            name = subject["name"]
            print(name)
            
            t1 = subject["t1"]
            t1ce = subject["t1ce"]
            t2 = subject["t2"]
            flair = subject["flair"]
            t1 : tio.Image
            t1ce : tio.Image
            t2 : tio.Image
            flair : tio.Image
            
            start, end = self.filter_slices(t1.data)
            t1.set_data(t1.data[..., start : end + 1])
            t1ce.set_data(t1ce.data[..., start : end + 1])
            t2.set_data(t2.data[..., start : end + 1])
            flair.set_data(flair.data[..., start : end + 1])

            t1_path, t1ce_path, t2_path, flair_path = [_img_path.format(grade, name, name) for _img_path in self._dst_paths]
            
            case_dir = os.path.split(t1_path)[0]
            os.makedirs(case_dir)
            t1.save(t1_path)
            t1ce.save(t1ce_path)
            t2.save(t2_path)
            flair.save(flair_path)

    def get_subjects(self):
        subjects = list()
        for case_name in self.case_names:
            t1_path, t1ce_path, t2_path, flair_path = [_img_path.format(case_name[1], case_name[0], case_name[0]) for _img_path in self._src_paths]
            
            subject = tio.Subject(
                t1=tio.ScalarImage(t1_path),
                t1ce=tio.ScalarImage(t1ce_path),
                t2=tio.ScalarImage(t2_path),
                flair=tio.ScalarImage(flair_path),
                name=case_name[0],
                grade=case_name[1]
            )
            subjects.append(subject) 
        return subjects
    
    def get_cases(self):     
        if isinstance(self.stage, list):
            case_names = list()
            for stage in self.stage:
                stage : STAGE
                stage_case_names = self.parse_txt(os.path.join(self.src_data_dir, "ImageSets/Main/" + stage.value +".txt"))
                case_names += stage_case_names
        else:
            case_names = self.parse_txt(os.path.join(self.src_data_dir, "ImageSets/Main/" + self.stage.value +".txt"))
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
