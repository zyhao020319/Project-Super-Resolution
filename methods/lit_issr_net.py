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
        self.sin: SliceInterpolationNet  # unet与stn层

        self.criterion = initialize_class(LocalCrossCorrelationWithSmoothnessLoss, cf.loss)
        self.criterion: LocalCrossCorrelationWithSmoothnessLoss

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
            loss=loss,
            ncc_loss=ncc_loss.detach(),
            smoothness_loss=smoothness_loss.detach()
        )

        return loss_dict

    def get_epoch_loss(self, batch_loss_dict_list: list):
        loss_list_dict = collections.defaultdict(list)
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

    def training_step(self, batch, batch_idx):
        moving_image, fixed_image = batch
        loss_dict = self.get_batch_loss(moving_image, fixed_image, batch_idx)

        outstream = ('[Epoch %d][Iter %d]'
                     '[loss: total %.2f || ncc %.2f || smoothness %.2f ]'
                     % (self.current_epoch,
                        batch_idx,
                        loss_dict["loss"],
                        loss_dict["ncc_loss"],
                        loss_dict["smoothness_loss"]))
        print(outstream, flush=True)

        return loss_dict

    def training_epoch_end(self, outputs):
        loss_dict = self.get_epoch_loss(outputs)
        cost_time = time.time() - self.timstamp

        tensorboard = self.logger.experiment
        tensorboard: SummaryWriter
        tensorboard.add_scalars(main_tag="train", tag_scalar_dict=loss_dict, global_step=self.current_epoch)

        outstream = ('[Epoch %d] [Train]'
                     '[loss: total %.2f || ncc %.2f || smoothness %.2f || time: %.2f]'
                     % (self.current_epoch,
                        loss_dict["loss"],
                        loss_dict["ncc_loss"],
                        loss_dict["smoothness_loss"],
                        cost_time))
        print(outstream, flush=True)

    def on_validation_epoch_start(self):
        self.timstamp = time.time()

    def validation_step(self, batch, batch_idx):
        tensorboard = self.logger.experiment
        tensorboard: SummaryWriter

        moving_image, fixed_image = batch
        loss_dict = self.get_batch_loss(moving_image, fixed_image, batch_idx)

        warped_moving_image, forward_dvf = self.sin.forward(moving_image, fixed_image)
        warped_fixed_image, backward_dvf = self.sin.forward(fixed_image, moving_image)

        forward_dvf_image = self.sin.stn.forward(self.grid, forward_dvf)
        backward_dvf_image = self.sin.stn.forward(self.grid, backward_dvf)

        rows = list()
        for moving_img, fixed_img, warped_moving_img, warped_fixed_img, forward_dvf_img, backward_dvf_img in zip(
                moving_image,
                fixed_image,
                warped_moving_image,
                warped_fixed_image,
                forward_dvf_image,
                backward_dvf_image):
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
                     '[loss: total %.2f || ncc %.2f || smoothness %.2f || time: %.2f]'
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
        self.create_grid(size, path, grid_density)
        grid_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        grid_image = cv2.resize(grid_image, dsize=(size[1], size[0]))[np.newaxis, np.newaxis, ...]
        grid_image: np.array
        grid_image = np.repeat(grid_image, batch_size, axis=0)
        return torch.from_numpy(grid_image).float()

    @staticmethod
    def create_grid(size, path, grid_density):
        num1, num2 = (size[0] + 10) // 10, (size[1] + 10) // grid_density  # 改变除数（10），即可改变网格的密度
        x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2), indexing='xy')

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
