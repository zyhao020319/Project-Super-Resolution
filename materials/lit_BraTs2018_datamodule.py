"""
    @Author : Yuan Dalong
    @Description : 实现BraTs2018数据集的加载器，训练和验证时加载切片，测试时加载volume
"""

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from materials.datasets.BraTs2018_slice import BraTs2018Slice as BraTs2018SliceDataset
from materials.datasets.BraTs2018_volume import BraTs2018Volume as BraTs2018VolumeDataset
from materials.utils import slice_collater, volume_collater


class LitBraTs2018Datamodule(pl.LightningDataModule):
    def __init__(self, sr_scale, batch_size, num_workers):  # 删除了num_workers = 4的初始化
        super().__init__()
        
        self.sr_scale    = sr_scale
        self.batch_size  = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.train_dataset = BraTs2018SliceDataset(sr_scale=self.sr_scale, stages=["train"])
        self.val_dataset   = BraTs2018SliceDataset(sr_scale=self.sr_scale, stages=["val"])
        self.test_dataset  = BraTs2018VolumeDataset(sr_scale=self.sr_scale, stages=["test"])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=slice_collater)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=slice_collater)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=volume_collater)
