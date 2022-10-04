import os
from enum import Enum


class BraTs2018SliceConfigs:
    def __init__(self):
        self.sr_scale = 8  # 超分辨率参数
        self.modality = "t1"

        self.batch_size = 4  # 批次大小
        self.num_workers = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])
        # 使用线程数（自动获取）


class MODALITY(Enum):
    T1 = "t1"
    T1CE = "t1ce"
    T2 = "t2"
    FLAIR = "flair"


class STAGE(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
