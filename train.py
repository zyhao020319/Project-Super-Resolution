import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks

from experiments.experiment_configs import ExperimentConfigs
from experiments.dataset_configs.brats2018slice_configs import BraTs2018SliceConfigs
from experiments.method_configs.issrnet_configs import ISSRNetConfigs
from materials.lit_BraTs2018_slice_loader import LitBraTs2018SliceLoader
from methods.lit_issr_net import LitISSRNet


def load_callbacks():  # 模型监控保存器
    callbacks = [pl_callbacks.EarlyStopping(  # 在模型评估阶段，模型效果如果没有提升，EarlyStopping 会通过设置 model.stop_training=True
        # 让模型提前停止训练
        monitor='val_loss',  # 监控量
        mode='min',  # 模式
        patience=20,  # 多少个epoch模型效果未提升会使模型提前停止训练
        min_delta=0.001  # 监控量最小改变值
    ), pl_callbacks.ModelCheckpoint(  # 回调类和model.fit联合使用，在训练阶段，保存模型权重和优化器状态信息
        monitor='val_loss',  # 要监视的指标
        filename='best-{epoch:02d}-{val_loss:.3f}',  # ckpt文件名
        save_top_k=1,  # 保存前k个最佳模型，k=-1的保存所有模型,k=0将不会保存模型，文件名后面会追加版本号，从v1开始
        mode='min',  # 监视指标的最大值还是最小值
        save_last=True  # 是否保存最后一次epoch训练的结果
    )]

    return callbacks


def issr_net_on_brats2018():
    exp_cf = ExperimentConfigs()  # 初始化实验参数
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=exp_cf.tb.save_dir, name=exp_cf.tb.exp_name)
    trainer = pl.Trainer(logger=tb_logger,  # 设置日志记录器
                         enable_progress_bar=True,  # 是否显示进度条
                         max_epochs=exp_cf.trainer.max_epochs,  # 最多训练轮数
                         # gpus=exp_cf.trainer.gpus,
                         accelerator=exp_cf.trainer.accelerator,  # 训练设备类
                         devices=exp_cf.trainer.devices,  # 训练设备数
                         num_sanity_val_steps=1,  # 开始训练前加载n个验证数据进行测试
                         callbacks=load_callbacks(),  # 添加回调函数或回调函数列表
                         #  overfit_batches=1
                         #  fast_dev_run=1
                         )

    brats2018slice_cf = BraTs2018SliceConfigs()  # 数据集参数
    lit_brats2018slice_loader = LitBraTs2018SliceLoader(cf=brats2018slice_cf)  # 数据集加载器

    issrnet_cf = ISSRNetConfigs()  # 形变场参数
    issrnet_cf.vis.batch_size = brats2018slice_cf.batch_size
    lit_issr_net = LitISSRNet(cf=issrnet_cf)

    trainer.fit(model=lit_issr_net,
                datamodule=lit_brats2018slice_loader
                )


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
