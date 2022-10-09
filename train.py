import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks

from experiments.experiment_configs import ExperimentConfigs
from experiments.dataset_configs.brats2018slice_configs import BraTs2018SliceConfigs
from experiments.method_configs.issrnet_configs import ISSRNetConfigs
from materials.lit_BraTs2018_slice_loader import LitBraTs2018SliceLoader
from methods.lit_issr_net import LitISSRNet


def load_callbacks():
    callbacks = [pl_callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=20,
        min_delta=0.001
    ), pl_callbacks.ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )]

    return callbacks


def issr_net_on_brats2018():
    exp_cf = ExperimentConfigs()
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=exp_cf.tb.save_dir, name=exp_cf.tb.exp_name)
    trainer = pl.Trainer(logger=tb_logger,
                         enable_progress_bar=False,
                         max_epochs=exp_cf.trainer.max_epochs,
                         # gpus=exp_cf.trainer.gpus,
                         accelerator=exp_cf.trainer.accelerator,
                         devices=exp_cf.trainer.devices,
                         num_sanity_val_steps=1,
                         callbacks=load_callbacks(),
                         #  overfit_batches=1
                         #  fast_dev_run=1
                         )

    brats2018slice_cf = BraTs2018SliceConfigs()
    lit_brats2018slice_loader = LitBraTs2018SliceLoader(cf=brats2018slice_cf)

    issrnet_cf = ISSRNetConfigs()
    issrnet_cf.vis.batch_size = brats2018slice_cf.batch_size
    lit_issr_net = LitISSRNet(cf=issrnet_cf)

    trainer.fit(model=lit_issr_net, datamodule=lit_brats2018slice_loader)


def voxelmorph_on_brats2018():
    from methods.lit_voxelmorph import LitVoxelMorph

    exp_cf = ExperimentConfigs()
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=exp_cf.tb.save_dir, name=exp_cf.tb.exp_name)
    trainer = pl.Trainer(logger=tb_logger,
                         enable_progress_bar=False,
                         max_epochs=exp_cf.trainer.max_epochs,
                         gpus=exp_cf.trainer.gpus,
                         num_sanity_val_steps=1,
                         callbacks=load_callbacks(),
                         #  overfit_batches=1
                         #  fast_dev_run=1
                         )

    brats2018slice_cf = BraTs2018SliceConfigs()
    datamodule = LitBraTs2018SliceLoader(cf=brats2018slice_cf)

    # args = Args()
    model = LitVoxelMorph()

    trainer.fit(model=model, datamodule=datamodule)


def voxelmorph_on_brats2018_0():
    from experiments.ours import voxelmorph_0 as cf
    from methods.lit_voxelmorph import LitVoxelMorph
    exp_cf = cf.exp
    datamodule_cf = cf.datamodule
    model_cf = cf.model

    trainer = pl.Trainer(**exp_cf.trainer)
    datamodule = LitBraTs2018SliceLoader(cf=datamodule_cf)
    model = LitVoxelMorph(cf=model_cf)

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    voxelmorph_on_brats2018_0()
