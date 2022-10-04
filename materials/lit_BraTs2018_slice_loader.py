from torch.utils.data import DataLoader
import pytorch_lightning as pl

from experiments.dataset_configs.brats2018slice_configs import BraTs2018SliceConfigs
from materials.datasets.BraTs2018_slice import BraTs2018DatasetSlice, STAGE
from materials.datasets.BraTs2018_volume import BraTs2018Volume as BraTs2018DatasetVolume


class LitBraTs2018SliceLoader(pl.LightningDataModule):
    def __init__(self, cf: BraTs2018SliceConfigs):
        super().__init__()
        
        self.sr_scale    = cf.sr_scale
        self.batch_size  = cf.batch_size
        self.num_workers = cf.num_workers
        
    def setup(self, stage=None):
        self.train_dataset = BraTs2018DatasetSlice(sr_scale=self.sr_scale, stage=STAGE.TRAIN)
        self.val_dataset   = BraTs2018DatasetSlice(sr_scale=self.sr_scale, stage=STAGE.VAL)
        self.test_dataset  = BraTs2018DatasetVolume(sr_scale=self.sr_scale, stage=STAGE.TEST)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)