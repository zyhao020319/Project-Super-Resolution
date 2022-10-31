import math
import time
import collections

from easydict import EasyDict
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch import optim
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from experiments.method_configs.issrnet_configs import ISSRNetConfigs
from methods.networks.slice_interpolation_net import SliceInterpolationNet
from methods.utils import initialize_class, normalize
from methods.losses.voxelmorph_loss import LocalCrossCorrelationWithSmoothnessLoss


class LitISSRNet(pl.LightningModule):
    def __init__(self, cf: ISSRNetConfigs, ):
        super().__init__()
        self.cf = cf

        self.sin = initialize_class(SliceInterpolationNet, cf.sin)
        self.sin: SliceInterpolationNet  # unet与stn层，输出预测图像和形变场

        self.criterion = initialize_class(LocalCrossCorrelationWithSmoothnessLoss, cf.loss)
        self.criterion: LocalCrossCorrelationWithSmoothnessLoss  # 实例化loss

        self.timstamp = 0

        grid = self.preparation(size=cf.vis["size"],
                                path=cf.vis["path"],
                                grid_density=cf.vis["grid_density"],
                                batch_size=cf.vis["batch_size"])
        self.register_buffer("grid", grid)

    def forward(self, moving_image: torch.Tensor, fixed_image: torch.Tensor):
        warped_image, dvf = self.sin.forward(moving_image, fixed_image)
        return warped_image, dvf

    def get_batch_loss(self, moving_image: torch.Tensor, fixed_image: torch.Tensor, batch_idx: int):
        warped_image, dvf = self.forward(moving_image, fixed_image)
        loss, ncc_loss, smoothness_loss = self.criterion.forward(warped_image, fixed_image, dvf)

        loss_dict = dict(
            loss=loss,  # 损失
            ncc_loss=ncc_loss.detach(),  # 相似损失，取消梯度运算
            smoothness_loss=smoothness_loss.detach()  # 平滑损失，取消梯度运算
        )

        return loss_dict

    def get_epoch_loss(self, batch_loss_dict_list: list):
        loss_list_dict = collections.defaultdict(list)  # 创建默认key的dict，使dict与list一样可以通过索引值索引
        for batch_loss_dict in batch_loss_dict_list:
            batch_loss_dict: dict
            for loss_name, loss_value in batch_loss_dict.items():
                loss_list_dict[loss_name].append(loss_value)

        mean_loss_dict = dict()
        for loss_name, loss_list in loss_list_dict.items():
            mean_loss_dict[loss_name] = sum(loss_list) / len(loss_list)

        return mean_loss_dict

    def on_train_epoch_start(self):
        self.timstamp = time.time()

    def training_step(self, batch, batch_idx):  # 重写step方法
        """
        在这里，您计算并返回训练损失和一些额外的度量，例如进度条或记录器。
        在此步骤中，您通常执行前向传递并计算批处理的损失。
        你也可以做一些更花哨的事情，比如多次向前传递或一些特定于模型的东西。

        例子:
            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss

        如果定义了多个优化器，则将使用附加的optimizer_idx参数调用此步骤。
        如果添加经过时间截断的反向传播，还将获得一个附加参数，其中包含前一步的隐藏状态。
        请注意 进度条中显示的损失值是对上一个值进行平滑(平均)处理的，因此它与训练/验证步骤中返回的实际损失不同。
        """
        moving_image, fixed_image = batch
        loss_dict = self.get_batch_loss(moving_image, fixed_image, batch_idx)

        outstream = ('[Epoch %d][Iter %d]'
                     '[loss: total %.8f || ncc %.8f || smoothness %.8f ]'
                     % (self.current_epoch,
                        batch_idx,
                        loss_dict["loss"],
                        loss_dict["ncc_loss"],
                        loss_dict["smoothness_loss"]))
        print(outstream, flush=True)

        return loss_dict

    def training_epoch_end(self, outputs):
        """
        在训练阶段结束时调用所有训练步骤的输出。如果需要对training_step返回的所有输出执行某些操作，则使用此方法
        """
        loss_dict = self.get_epoch_loss(outputs)  # 计算epoch损失
        cost_time = time.time() - self.timstamp  # 计时

        tensorboard = self.logger.experiment
        tensorboard: SummaryWriter
        tensorboard.add_scalars(main_tag="train", tag_scalar_dict=loss_dict, global_step=self.current_epoch)

        outstream = ('[Epoch %d] [Train]'
                     '[loss: total %.8f || ncc %.8f || smoothness %.8f || time: %.2f]'
                     % (self.current_epoch,
                        loss_dict["loss"],
                        loss_dict["ncc_loss"],
                        loss_dict["smoothness_loss"],
                        cost_time))
        print(outstream, flush=True)

    def on_validation_epoch_start(self):  # 训练开始时初始化计时器
        self.timstamp = time.time()

    def validation_step(self, batch, batch_idx):
        """
        对来自验证集的单个批数据进行操作。在这一步中，您可能会生成示例或计算任何感兴趣的内容，如准确性。
        """
        tensorboard = self.logger.experiment
        tensorboard: SummaryWriter

        moving_image, fixed_image = batch
        loss_dict = self.get_batch_loss(moving_image, fixed_image, batch_idx)

        warped_moving_image, forward_dvf = self.sin.forward(moving_image, fixed_image)
        warped_fixed_image, backward_dvf = self.sin.forward(fixed_image, moving_image)  # 获取转换后的图层和转换图层

        forward_dvf_image = self.sin.stn.forward(self.grid, forward_dvf)
        backward_dvf_image = self.sin.stn.forward(self.grid, backward_dvf)

        rows = list()
        for moving_img, fixed_img, warped_moving_img, warped_fixed_img, forward_dvf_img, backward_dvf_img in zip(
                moving_image,
                fixed_image,
                warped_moving_image,
                warped_fixed_image,
                forward_dvf_image,
                backward_dvf_image):  # 创建一个新对象，length为img的length
            displayed_images = normalize(warped_fixed_img, moving_img)[::-1] + normalize(warped_moving_img, fixed_img)[
                                                                               ::-1] + [
                                   torch.clamp(forward_dvf_img, 0, 255).type(dtype=torch.uint8),
                                   torch.clamp(backward_dvf_img, 0, 255).type(dtype=torch.uint8),
                               ]
            row = torchvision.utils.make_grid(displayed_images, nrow=6, pad_value=255)
            rows.append(row)
        grid = torchvision.utils.make_grid(rows, nrow=1, padding=4, pad_value=255)
        tensorboard.add_image("orig_warped_dvf", grid, global_step=self.current_epoch)

        return loss_dict

    def validation_epoch_end(self, outputs):
        loss_dict = self.get_epoch_loss(outputs)
        cost_time = time.time() - self.timstamp

        self.log("val_loss", loss_dict["loss"], logger=False, on_step=False, on_epoch=True)

        tensorboard = self.logger.experiment
        tensorboard: SummaryWriter
        tensorboard.add_scalars(main_tag="val", tag_scalar_dict=loss_dict, global_step=self.current_epoch)

        outstream = ('[Epoch %d] [Val]'
                     '[loss: total %.8f || ncc %.8f || smoothness %.8f || time: %.8f]'
                     % (self.current_epoch,
                        loss_dict["loss"],
                        loss_dict["ncc_loss"],
                        loss_dict["smoothness_loss"],
                        cost_time))
        print(outstream, flush=True)

    def configure_optimizers(self):
        adam_params = EasyDict(
            params=self.sin.parameters()
        )
        adam_params.update(self.cf.adam_optimizer)
        optimizer = initialize_class(optim.Adam, adam_params)
        optimizer: optim.Adam

        plateau_params = EasyDict(
            optimizer=optimizer
        )
        plateau_params.update(self.cf.plateau_scheduler)
        scheduler = initialize_class(optim.lr_scheduler.ReduceLROnPlateau, plateau_params)
        scheduler: optim.lr_scheduler.ReduceLROnPlateau

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    # -----------------------------------------------------------------#
    # vis util funcs    
    # -----------------------------------------------------------------#
    def preparation(self, size, path, grid_density, batch_size):
        self.create_grid(size, path, grid_density)  # 生成网格
        grid_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 以灰度读取图像，输出数组
        grid_image = cv2.resize(grid_image, dsize=(size[1], size[0]))[np.newaxis, np.newaxis, ...]  # resize并转换为np数组
        grid_image: np.array
        grid_image = np.repeat(grid_image, batch_size, axis=0)  # 在dim=0维度上重复batch_size次
        return torch.from_numpy(grid_image).float()

    @staticmethod
    def create_grid(size, path, grid_density):
        num1, num2 = (size[0] + 10) // 10, (size[1] + 10) // grid_density  # 改变除数（10），即可改变网格的密度
        x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2), indexing='xy')  # 生成网格

        plt.figure(figsize=((size[0] + 10) / 100.0, (size[1] + 10) / 100.0))  # 指定图像大小
        plt.plot(x, y, color="black")
        plt.plot(x.transpose(), y.transpose(), color="black")
        plt.axis('off')  # 不显示坐标轴
        # 去除白色边框
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(path)  # 保存图像
