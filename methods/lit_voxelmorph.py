import math
import time
import collections
import os

from easydict import EasyDict
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch import optim
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import tifffile

from experiments.method_configs.voxelmorph import VoxelMorphConfigs
from methods.networks.voxelmorph.networks import VxmDense
from methods.utils import normalize
from methods.networks.voxelmorph.losses import NCCWithGrad


class LitVoxelMorph(pl.LightningModule):
    def __init__(self, cf: VoxelMorphConfigs):
        super().__init__()
        self.cf = cf

        self.sin = VxmDense(**cf.vxm)
        self.criterion = NCCWithGrad(**cf.loss)

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
        loss, [ncc_loss, smoothness_loss] = self.criterion.forward([(warped_image, fixed_image), (dvf,)])

        loss_dict = dict(
            loss=loss,
            ncc_loss=ncc_loss,
            smoothness_loss=smoothness_loss
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

        forward_dvf_image = self.sin.transformer.forward(self.grid, forward_dvf)
        backward_dvf_image = self.sin.transformer.forward(self.grid, backward_dvf)

        rows = list()
        for moving_img, fixed_img, warped_moving_img, warped_fixed_img, forward_dvf_img, backward_dvf_img in zip(
                moving_image,
                fixed_image,
                warped_moving_image,
                warped_fixed_image,
                forward_dvf_image,
                backward_dvf_image):
            displayed_images = normalize(warped_fixed_img[0:1], moving_img[0:1])[::-1] + normalize(
                warped_moving_img[0:1], fixed_img[0:1])[::-1] + [
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

    def test_step(self, batch, batch_idx: int):
        lr_image, hr_image = batch
        lr_image: torch.Tensor
        hr_image: torch.Tensor

        slices = list()
        for i in range(lr_image.shape[-1] - 1):
            I0 = lr_image[..., i]
            I1 = lr_image[..., i + 1]
            I0: torch.Tensor
            I1: torch.Tensor

            _, F_1_0 = self.sin.forward(I0, I1)
            _, F_0_1 = self.sin.forward(I1, I0)

            validationFrameIndex = np.arange(7)
            fCoeff = self.getFlowCoeff(validationFrameIndex, self.device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            I0 = I0.expand(F_t_0.shape[0], *I0.shape[1:])
            I1 = I1.expand(F_t_1.shape[0], *I1.shape[1:])
            g_I0_F_t_0 = self.sin.transformer(I0, F_t_0)  # 根据估计出的t->0的形变场从frame 0预测出frame t
            g_I1_F_t_1 = self.sin.transformer(I1, F_t_1)  # 根据估计出的t->1的形变场从frame 1预测出frame t

            g = (g_I0_F_t_0 + g_I1_F_t_1) / 2
            I0 = I0[0].unsqueeze(0)
            I1 = I1[1].unsqueeze(0)

            slices.append(I0)
            slices.append(g)
        slices.append(lr_image[..., -1])

        pred_hr_image = torch.cat(slices)
        pd_hr_image = pred_hr_image.permute(1, 2, 3, 0)
        gt_hr_image = hr_image[0]
        gt_lr_image = lr_image[0]

        images = list()
        for image in [gt_lr_image, pd_hr_image, gt_hr_image]:
            image = (image + 1) / 2 * 32767
            image = torch.clamp(image, 0, 32767)
            image = image.type(torch.int16).detach().cpu().numpy()
            images.append(image)
        gt_lr_image, pd_hr_image, gt_hr_image = images

        modalities = ["t1", "t1ce", "t2", "flair"]

        case_save_dir = os.path.join(self.cf.test_output_save_dir, str(batch_idx))
        os.makedirs(os.path.join(case_save_dir))
        gt_lr_save_dir = os.path.join(case_save_dir, "gt_lr")
        pd_hr_save_dir = os.path.join(case_save_dir, "pd_hr")
        gt_hr_save_dir = os.path.join(case_save_dir, "gt_hr")

        for save_dir, fused_image in zip([gt_lr_save_dir, pd_hr_save_dir, gt_hr_save_dir],
                                         [gt_lr_image, pd_hr_image, gt_hr_image]):
            for modality, image in zip(modalities, fused_image):
                tifffile.imwrite(os.path.join(save_dir, modality + ".tiff"), image)

    @staticmethod
    def _normalize(tensors: torch.Tensor):
        min_ = min(torch.min(tensors[0]), torch.min(tensors[-1]))
        max_ = max(torch.max(tensors[0]), torch.max(tensors[-1]))

        tensors = torch.clamp(tensors, min=min_, max=max_)
        tensors = (tensors - min_) / (max_ - min_) * 255
        tensors = tensors.type(torch.uint8)

        return tensors

    def configure_optimizers(self):
        adam_params = EasyDict(
            params=self.sin.parameters()
        )
        adam_params.update(self.cf.adam_optimizer)
        optimizer = optim.Adam(**adam_params)

        cos_anneal_params = EasyDict(
            optimizer=optimizer
        )
        cos_anneal_params.update(self.cf.cos_anneal_scheduler)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(**cos_anneal_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int = 0, optimizer_closure=None,
                       on_tpu: bool = False, using_native_amp: bool = False, using_lbfgs: bool = False):
        if self.global_step < self.cf.lr_warmup.num_warmup_steps:
            lr_scale = min(1., float(self.global_step + 1) / self.cf.lr_warmup.num_warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.cf.adam_optimizer.lr
        # update params
        optimizer.step(closure=optimizer_closure)

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

    # ------------------------------------------------------------------------------#
    # 用于中间切片插值的一些函数，主要参考SuperSlomo
    # ------------------------------------------------------------------------------#
    @staticmethod
    def getFlowCoeff(ind, device):
        """
        Gets flow coefficients used for calculating intermediate optical
        flows from optical flows between I0 and I1: F_0_1 and F_1_0.

        F_t_0 = C00 x F_0_1 + C01 x F_1_0
        F_t_1 = C10 x F_0_1 + C11 x F_1_0

        where,
        C00 = -(1 - t) x t
        C01 = t x t
        C10 = (1 - t) x (1 - t)
        C11 = -t x (1 - t)

        Parameters
        ----------
            indices : tensor
                indices corresponding to the intermediate frame positions
                of all samples in the batch.
            device : device
                    computation device (cpu/cuda). 

        Returns
        -------
            tensor
                coefficients C00, C01, C10, C11.
        """
        t = np.linspace(0.125, 0.875, 7)
        # Convert indices tensor to numpy array
        C11 = C00 = - (1 - (t[ind])) * (t[ind])
        C01 = (t[ind]) * (t[ind])
        C10 = (1 - (t[ind])) * (1 - (t[ind]))
        return torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C01)[None, None,
                                                                                      None, :].permute(3, 0, 1, 2).to(
            device), torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C11)[None,
                                                                                            None, None, :].permute(3, 0,
                                                                                                                   1,
                                                                                                                   2).to(
            device)
