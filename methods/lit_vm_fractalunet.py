import enum
import math
import time
import collections
import os
import pandas as pd

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
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio  as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

from experiments.base_configs.methods.lit_vm_fractalunet import LitVMFractalUnetConfigs
from methods.networks.vm_fractalunet import VxmDenseFractalUnet
from methods.utils import normalize, stack_images
from methods.losses.voxelmorph import NCCWithGrad


class LitVMFractalUnet(pl.LightningModule):
    def __init__(self, cf: LitVMFractalUnetConfigs):
        super().__init__()
        self.cf = cf
        
        self.sin = VxmDenseFractalUnet(**cf.vxm)
        self.criterion = NCCWithGrad(**cf.loss)
        
        self.timstamp = 0
        
        grid = self.preparation(size=cf.vis["size"],
                                path=cf.vis["path"],
                                grid_density=cf.vis["grid_density"],
                                batch_size=cf.vis["batch_size"])
        self.register_buffer("grid", grid)
        
    def forward(self, slice_0: torch.Tensor, slice_1: torch.Tensor):
        warped_slice_0, forward_dvf = self.sin.forward(slice_0, slice_1)
        warped_slice_1, backward_dvf = self.sin.forward(slice_1, slice_0)
        return warped_slice_0, forward_dvf, warped_slice_1, backward_dvf
    
    def get_batch_loss(self, slice_0: torch.Tensor, slice_1 : torch.Tensor, are_selected : torch.Tensor, batch_idx: int):
        warped_slice_0, forward_dvf, warped_slice_1, backward_dvf = self.forward(slice_0, slice_1)
        loss, [ncc_loss, smoothness_loss] = self.criterion.forward([
            (warped_slice_0, slice_1, are_selected), 
            (warped_slice_1, slice_0, are_selected), 
            (forward_dvf,),
            (backward_dvf,)
        ])
        
        loss_dict = dict(
            loss = loss,
            ncc_loss = ncc_loss,
            smoothness_loss = smoothness_loss
        )
        
        return loss_dict

    def get_epoch_loss(self, batch_loss_dict_list : list):
        loss_list_dict = collections.defaultdict(list)
        for batch_loss_dict in batch_loss_dict_list:
            batch_loss_dict : dict
            for loss_name, loss_value in batch_loss_dict.items():
                loss_list_dict[loss_name].append(loss_value)
        
        mean_loss_dict = dict()
        for loss_name, loss_list in loss_list_dict.items():
            mean_loss_dict[loss_name] = sum(loss_list) / len(loss_list)
        
        return mean_loss_dict   
    
    def on_train_epoch_start(self):
        self.timstamp = time.time()   
    
    def training_step(self, batch, batch_idx):
        slice_0, slice_1, are_selected = batch
        loss_dict = self.get_batch_loss(slice_0, slice_1, are_selected, batch_idx)

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
        tensorboard : SummaryWriter 
        tensorboard.add_scalars(main_tag="train", tag_scalar_dict=loss_dict, global_step=self.current_epoch)

        outstream = ('[Epoch %d] [Train]'
                '[loss: total %.2f || ncc %.2f || smoothness %.2f || time: %.2f]'
                % (self.current_epoch,  
                    loss_dict["loss"],
                    loss_dict["ncc_loss"],
                    loss_dict["smoothness_loss"],
                    cost_time))
        print(outstream, flush=True)               
    
    def validation_step(self, batch, batch_idx):
        tensorboard = self.logger.experiment
        tensorboard : SummaryWriter

        slice_0, slice_1, are_selected = batch
        loss_dict = self.get_batch_loss(slice_0, slice_1, are_selected, batch_idx)
        
        warped_slice_0, forward_dvf, warped_slice_1, backward_dvf = self.forward(slice_0, slice_1)
        
        forward_dvf_image  = self.sin.transformer.forward(self.grid, forward_dvf)
        backward_dvf_image = self.sin.transformer.forward(self.grid, backward_dvf)
        
        rows = list()
        for _slice_0, _slice_1, _warped_slice_0, _warped_slice_1, _forward_dvf_img, _backward_dvf_img, _are_selected in zip(
            slice_0,
            slice_1,
            warped_slice_0,
            warped_slice_1,
            forward_dvf_image,
            backward_dvf_image,
            are_selected
        ):
            displayed_images = normalize(_warped_slice_1[_are_selected][0:1], _slice_0[_are_selected][0:1])[::-1] + normalize(_warped_slice_0[_are_selected][0:1], _slice_1[_are_selected][0:1])[::-1] + [
                torch.clamp(_forward_dvf_img, 0, 255).type(dtype=torch.uint8),
                torch.clamp(_backward_dvf_img, 0, 255).type(dtype=torch.uint8),
            ]
            row = torchvision.utils.make_grid(displayed_images, nrow=6, pad_value=255)
            rows.append(row)
        grid = torchvision.utils.make_grid(rows, nrow=1, padding=4, pad_value=255)
        tensorboard.add_image("orig_warped_dvf", grid, global_step=self.current_epoch)
            
        return loss_dict
    
    def validation_epoch_end(self, outputs):
        loss_dict = self.get_epoch_loss(outputs)
        
        self.log("val_loss", loss_dict["loss"], logger=False, on_step=False, on_epoch=True)
        
        tensorboard = self.logger.experiment
        tensorboard : SummaryWriter
        tensorboard.add_scalars(main_tag="val", tag_scalar_dict=loss_dict, global_step=self.current_epoch)

        outstream = ('[Epoch %d] [Val]'
                '[loss: total %.2f || ncc %.2f || smoothness %.2f]'
                % (self.current_epoch,  
                    loss_dict["loss"],
                    loss_dict["ncc_loss"],
                    loss_dict["smoothness_loss"]
                    ))
        print(outstream, flush=True)
        
    def on_test_start(self):
        self.mses = [collections.defaultdict(list) for _ in range(3)]
        self.psnrs = [collections.defaultdict(list) for _ in range(3)]
        self.ssims = [collections.defaultdict(list) for _ in range(3)]
        
    def test_step(self, batch, batch_idx: int):
        lr_image, hr_image = batch
        lr_image : torch.Tensor
        hr_image : torch.Tensor
        
        slices = list()
        mid_slices = list()
        for i in range(lr_image.shape[-1] - 1):
            I0 = lr_image[..., i]
            I1 = lr_image[..., i + 1]
            I0 : torch.Tensor
            I1 : torch.Tensor
            
            _, F_1_0 = self.sin.forward(I0, I1)
            _, F_0_1 = self.sin.forward(I1, I0)

            validationFrameIndex = np.arange(7)
            fCoeff = self.getFlowCoeff(validationFrameIndex, self.device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0        

            I0 = I0.expand(F_t_0.shape[0], *I0.shape[1:])
            I1 = I1.expand(F_t_1.shape[0], *I1.shape[1:])
            g_I0_F_t_0 = self.sin.transformer(I0, F_t_0) # 根据估计出的t->0的形变场从frame 0预测出frame t
            g_I1_F_t_1 = self.sin.transformer(I1, F_t_1) # 根据估计出的t->1的形变场从frame 1预测出frame t
            
            w_t_0 = torch.linspace(0.875, 0.125, 7, device=self.device).reshape(7, 1, 1, 1)
            w_t_1 = torch.linspace(0.125, 0.875, 7, device=self.device).reshape(7, 1, 1, 1)
            g = g_I0_F_t_0 * w_t_0 + g_I1_F_t_1 * w_t_1
            I0 = I0[0].unsqueeze(0)
            I1 = I1[1].unsqueeze(0)
            
            slices.append(I0)
            slices.append(g)
            
            if i == lr_image.shape[-1] // 2:
                mid_slices.append(I0)
                mid_slices.append(g)
                mid_slices.append(I1)
        slices.append(lr_image[..., -1])

        #---------------------------------------------------------------------#
        # 对低分辨率的中间两个切片的插值效果进行可视化
        #---------------------------------------------------------------------#
        modalities = ["t1", "t1ce", "t2", "flair"]
        multimodal_mid_slices = torch.cat(mid_slices)
        multimodal_mid_slices_dir = os.path.join(self.cf.test_output_save_dir, "mid_slices")
        os.makedirs(multimodal_mid_slices_dir, exist_ok=True)
        for i in range(4):
            modality = modalities[i]
            mid_slices = multimodal_mid_slices[::2, i:i+1, ...]
            mid_slices = self._normalize(mid_slices)
            mid_slices = torchvision.utils.make_grid(mid_slices, nrow=5, padding=2, pad_value=255)
        
            mid_slices_dir = os.path.join(multimodal_mid_slices_dir, modality)
            os.makedirs(mid_slices_dir, exist_ok=True)
            torchvision.utils.save_image(mid_slices / 255, os.path.join(mid_slices_dir, str(batch_idx) + ".png"))

        #---------------------------------------------------------------------#
        # 存储三视图
        #---------------------------------------------------------------------#
        li_lr_image = F.interpolate(lr_image, size=hr_image.shape[-3:], mode="trilinear", align_corners=True)
        ni_lr_image = F.interpolate(lr_image, size=hr_image.shape[-3:], mode="nearest")
        pred_hr_image = torch.cat(slices)
        
        pd_hr_image = pred_hr_image.permute(1, 2, 3, 0)
        gt_hr_image = hr_image[0]
        # ni_lr_image = ni_lr_image[0]
        # li_lr_image = li_lr_image[0]
        
        multimodal_ni_lr_views = self._three_views(ni_lr_image) # 经过最近邻插值的低分辨率图像
        multimodal_li_lr_views = self._three_views(li_lr_image) # 经过线性插值的低分辨率图像
        multimodal_pd_hr_views = self._three_views(pd_hr_image) # 
        multimodal_gt_hr_views = self._three_views(gt_hr_image)
        
        multimodal_three_views_dir = os.path.join(self.cf.test_output_save_dir, "three_views")
        os.makedirs(multimodal_three_views_dir, exist_ok=True)        
        for i in range(4):
            modality = modalities[i]
            ni_lr_views = multimodal_ni_lr_views[:, i:i+1, ...]
            li_lr_views = multimodal_li_lr_views[:, i:i+1, ...]
            pd_hr_views = multimodal_pd_hr_views[:, i:i+1, ...]
            gt_hr_views = multimodal_gt_hr_views[:, i:i+1, ...]
            
            views_0 = torch.stack([ni_lr_views[0], li_lr_views[0], pd_hr_views[0], gt_hr_views[0]])
            views_1 = torch.stack([ni_lr_views[1], li_lr_views[1], pd_hr_views[1], gt_hr_views[1]])
            views_2 = torch.stack([ni_lr_views[2], li_lr_views[2], pd_hr_views[2], gt_hr_views[2]])
            
            views_0 = self._normalize(views_0)
            views_1 = self._normalize(views_1)
            views_2 = self._normalize(views_2)
            
            views_0 = torchvision.utils.make_grid(views_0, nrow=4, padding=3, pad_value=255)
            views_1 = torchvision.utils.make_grid(views_1, nrow=4, padding=3, pad_value=255)
            views_2 = torchvision.utils.make_grid(views_2, nrow=4, padding=3, pad_value=255)
            
            all_views = torchvision.utils.make_grid([views_0, views_1, views_2], nrow=1, padding=2,  pad_value=255)

            three_views_dir = os.path.join(multimodal_three_views_dir, modality) 
            os.makedirs(three_views_dir, exist_ok=True)           
            torchvision.utils.save_image(all_views / 255, os.path.join(three_views_dir, str(batch_idx) + ".png"))

        images = list()
        for image in [ni_lr_image, li_lr_image, pd_hr_image, gt_hr_image]:
            print(image.shape)
            image = (image + 1) / 2 * 32767
            image = torch.clamp(image, 0, 32767)
            image = image.type(torch.int16).detach().cpu().numpy()
            images.append(image)
        print(np.all(images[0] == images[-1]))
        print(np.all(images[1] == images[-1]))
        
        for i, image in enumerate(images[:-1]):
            for j, (image_test, image_true) in enumerate(zip(image, images[-1])):
                modality = modalities[j]
                print(np.all(image_true == image_test))
                self.mses[i][modality].append(mse(image_true, image_test))
                self.ssims[i][modality].append(ssim(image_true, image_test))
                dr = np.max([image_true.max(), image_test.max()]) - np.min([image_true.min(), image_test.min()])                 
                self.psnrs[i][modality].append(psnr(image_true, image_test, data_range=dr))

    def test_epoch_end(self, outputs):
        sr_method_names = ["nearest_interpolation", "linear_interpolation", "dvf_interpolation"]
        metrics_dir = os.path.join(self.cf.test_output_save_dir, "metrics")
        for i, sr_method in enumerate(sr_method_names):
            sr_method_dir = os.path.join(metrics_dir, sr_method)
            os.makedirs(sr_method_dir, exist_ok=True)
            mses = pd.DataFrame(data=self.mses[i])
            psnrs = pd.DataFrame(data=self.psnrs[i])
            ssims = pd.DataFrame(data=self.ssims[i])
            mses.to_csv(os.path.join(sr_method_dir, "mses.csv"))
            psnrs.to_csv(os.path.join(sr_method_dir, "psnrs.csv"))
            ssims.to_csv(os.path.join(sr_method_dir, "ssims.csv"))

    @staticmethod
    def _normalize(tensors : torch.Tensor):
        min_ = min(torch.min(tensors[0]), torch.min(tensors[-1]))
        max_ = max(torch.max(tensors[0]), torch.max(tensors[-1]))
        
        tensors = torch.clamp(tensors, min=min_, max=max_)
        tensors = (tensors  - min_) / (max_ - min_) * 255
        tensors = tensors.type(torch.uint8)
        
        return tensors
    
    @staticmethod
    def _three_views(volume : torch.Tensor):
        height, width, depth = volume.shape[-3:]
        slice_0 = volume[..., depth // 2]
        slice_1 = volume[..., width // 2, :]
        slice_2 = volume[:, height // 2, ...]
        views = [slice_0, slice_1, slice_2]
        
        views = stack_images(views)
        return views        
        
    def configure_optimizers(self):
        adam_params = EasyDict(
            params = self.sin.parameters()
        )
        adam_params.update(self.cf.adam_optimizer)
        optimizer = optim.Adam(**adam_params)

        cos_anneal_params = EasyDict(
            optimizer = optimizer
        )
        cos_anneal_params.update(self.cf.cos_anneal_scheduler)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(**cos_anneal_params)
        
        return {
            "optimizer" : optimizer,
            "lr_scheduler" : scheduler
        }

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int = 0, optimizer_closure = None, on_tpu: bool = False, using_native_amp: bool = False, using_lbfgs: bool = False):
        if self.global_step < self.cf.lr_warmup.num_warmup_steps:
            lr_scale = min(1., float(self.global_step + 1) / self.cf.lr_warmup.num_warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.cf.adam_optimizer.lr
        # update params
        optimizer.step(closure=optimizer_closure)
   
    #-----------------------------------------------------------------#
    # vis util funcs    
    #-----------------------------------------------------------------#    
    def preparation(self, size, path, grid_density, batch_size):
        self.create_grid(size, path, grid_density)
        grid_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        grid_image = cv2.resize(grid_image, dsize=(size[1], size[0]))[np.newaxis, np.newaxis, ...]
        grid_image : np.array
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

    #------------------------------------------------------------------------------#
    # 用于中间切片插值的一些函数，主要参考SuperSlomo
    #------------------------------------------------------------------------------#
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
        return torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C01)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C11)[None, None, None, :].permute(3, 0, 1, 2).to(device)

    