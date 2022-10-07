from torch.utils.data import DataLoader
import pytorch_lightning as pl

from materials.datasets.BraTs2018_volume import BraTs2018Volume
from experiments.dataset_configs.brats2018slice_configs import STAGE, MODALITY


str2modality = {
    "t1" : MODALITY.T1,
    "t1ce" : MODALITY.T1CE,
    "t2" : MODALITY.T2,
    "flair" : MODALITY.FLAIR
}


class LitISSRLoader(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, modality: str):
        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.modality = str2modality[modality]
        
    def setup(self, stage=None):
        self.train_dataset = BraTs2018Volume(modality=self.modality, stage=[STAGE.TRAIN, STAGE.VAL])
        self.val_dataset = BraTs2018Volume(modality=self.modality, stage=STAGE.TEST)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)
        