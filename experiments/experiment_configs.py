import os

from easydict import EasyDict
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers


class Callbacks:
    def __init__(self):
        self.early_stopping = pl_callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=50,
            min_delta=1e-4
        )

        self.save_checkpoint = pl_callbacks.ModelCheckpoint(
            monitor='val_loss',
            filename='best-{epoch:02d}-{val_loss:.3f}',
            save_top_k=1,
            mode='min',
            save_last=True
        )

        self.log_learning_rate = pl_callbacks.LearningRateMonitor(
            logging_interval="step"
        )
        
    def __call__(self):
        return [self.early_stopping, self.save_checkpoint, self.log_learning_rate]

class ExperimentConfigs:
    def __init__(self, exp_name: str, save_dir="results/ours", test_output_save_dir=None):
        os.makedirs(os.path.join(save_dir, exp_name), exist_ok=True)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir, exp_name)

        if test_output_save_dir is None:
            pass
        else:
            self.test_output_path = os.path.join(test_output_save_dir, "test_output")
            os.makedirs(self.test_output_path, exist_ok=True)

        self.trainer = EasyDict(
            logger = tb_logger,
            max_epochs = 1000,
            gpus = 1,
            num_nodes = 1,
            strategy = None,
            resume_from_checkpoint = None,
            # num_sanity_val_steps = 3,
            # fast_dev_run = 1,
            # overfit_batches = 1,
            enable_progress_bar = False,
            callbacks = Callbacks()()
        )
