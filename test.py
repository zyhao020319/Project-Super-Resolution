import pytorch_lightning as pl
from materials.lit_BraTs2018_slice_loader import LitBraTs2018SliceLoader


def voxelmorph_on_brats2018_0():
    from experiments.ours import voxelmorph_0 as cf
    from methods.lit_voxelmorph import LitVoxelMorph
    exp_cf = cf.exp
    datamodule_cf = cf.datamodule
    model_cf = cf.model

    trainer = pl.Trainer(**exp_cf.trainer)
    datamodule = LitBraTs2018SliceLoader(cf=datamodule_cf)
    model = LitVoxelMorph.load_from_checkpoint(
        checkpoint_path="results/ours/voxelmorph/version_0/checkpoints/best-epoch=334-val_loss=-0.115.ckpt",
        cf=model_cf)

    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    voxelmorph_on_brats2018_0()
