import time
import inspect

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from torch.utils.tensorboard import SummaryWriter

from methods.networks import ISSRGenerator, ISSRDiscriminator


class LitISSRGAN(pl.LightningModule):
    def __init__(self, model_params: dict, optimizer_params: dict, loss_params: dict, extra_params: dict):
        super().__init__()
        self.save_hyperparameters()
        
        self.gen = self.initialize_model(ISSRGenerator, model_params=model_params["generator"])
        self.dis = self.initialize_model(ISSRDiscriminator, model_params=model_params["discriminator"])
        
        real_label = torch.ones(extra_params["dis_out_shape"], dtype=torch.float32)
        fake_label = torch.zeros(extra_params["dis_out_shape"], dtype=torch.float32)
        self.register_buffer("real_label", real_label)
        self.register_buffer("fake_label", fake_label)
        
        self.l1_loss_weight = self.hparams.loss_params["l1_loss_weight"]
        self.num_sample_slices = self.hparams.loss_params["num_sample_slices"]
        
        self.epoch_idx = 0
        
    def forward(self, x: torch.Tensor):
        return self.gen(x)
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        tensorboard = self.logger.experiment
        tensorboard : SummaryWriter
        islr_imgs, _, _ = batch

        # train generator
        if optimizer_idx == 0:
            # generate images
            t0 = time.time()
            issr_imgs = self(islr_imgs)

            # adversarial loss is binary cross-entropy
            g_l1_loss = self.l1_loss(issr_imgs, islr_imgs)
            coronal_and_sagittal_sections = self.sample_slices(issr_imgs, is_real=False)
            g_gan_loss = self.adversarial_loss(self.dis(coronal_and_sagittal_sections), is_real=False)
            g_loss = g_gan_loss + self.l1_loss_weight * g_l1_loss
            
            t1 = time.time()
            cost_time = t1 - t0
            
            loss_dict = dict(
                g_loss=g_loss.detach(),
                g_l1_loss=g_l1_loss.detach(),
                g_gan_loss=g_gan_loss.detach(),
                time=cost_time
            )
            outstream = ('[Epoch %d][Iter %d]'
                    '[G_loss: total %.2f || l1 %.2f || gan %.2f || time: %.2f]'
                    % (self.epoch_idx,  
                        batch_idx, 
                        loss_dict['g_loss'].item(),
                        loss_dict['g_l1_loss'].item(),
                        loss_dict['g_gan_loss'].item(), 
                        cost_time))
            print(outstream, flush=True)
            # self.log_dict(loss_dict, prog_bar=True)
            tensorboard.add_scalars(main_tag="G", tag_scalar_dict=loss_dict, global_step=batch_idx)
            
            displayed_sections = coronal_and_sagittal_sections[:6]
            grid = torchvision.utils.make_grid(displayed_sections)
            tensorboard.add_image(tag="train_sr_mri_coronal_and_sagittal_sections", img_tensor=grid, global_step=batch_idx)
            
            return {"loss": g_loss, "time": cost_time}

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
 
            # how well can it label as real?
            t0 = time.time()
            transverse_sections = self.sample_slices(islr_imgs, is_real=True)
            d_real_loss = self.adversarial_loss(self.dis(transverse_sections), is_real=True)

            # how well can it label as fake?
            issr_imgs = self(islr_imgs)
            coronal_and_sagittal_sections = self.sample_slices(issr_imgs, is_real=False)
            d_fake_loss = self.adversarial_loss(self.dis(coronal_and_sagittal_sections), is_real=False)

            # discriminator loss is the average of these
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            t1 = time.time()
            cost_time = t1 - t0
            
            loss_dict = dict(
                d_loss=d_loss.detach(),
                d_real_loss=d_real_loss.detach(),
                d_fake_loss=d_fake_loss.detach()
            )
            outstream = ('[Epoch %d][Iter %d]'
                    '[D_loss: total %.2f || real %.2f || fake %.2f || time: %.2f]'
                    % (self.epoch_idx,  
                        batch_idx, 
                        loss_dict['d_loss'].item(),
                        loss_dict['d_real_loss'].item(),
                        loss_dict['d_fake_loss'].item(), 
                        cost_time))
            print(outstream, flush=True) 
            self.log("time", cost_time, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            tensorboard.add_scalars(main_tag="D", tag_scalar_dict=loss_dict, global_step=batch_idx)
                      
            return {"loss": d_loss, "time": cost_time}
    
    def training_epoch_end(self, outputs):
        self.epoch_idx += 1 
    
    def validation_step(self, batch, batch_idx):
        tensorboard = self.logger.experiment
        tensorboard : SummaryWriter
        
        islr_imgs, gt_imgs, _ = batch 
        issr_imgs = self(islr_imgs)
        merged_sections = self.validation_comparison_util(issr_imgs=issr_imgs, gt_imgs=gt_imgs, num_displayed_slices=6)
        grid = torchvision.utils.make_grid(merged_sections)
        tensorboard.add_image(tag="val_sr_mri_coronal_and_sagittal_sections", img_tensor=grid, global_step=batch_idx)           
        
    @staticmethod
    def validation_comparison_util(issr_imgs: torch.Tensor, gt_imgs: torch.Tensor, num_displayed_slices: int):
        assert np.all(issr_imgs.shape == gt_imgs.shape)
        batch_size, in_channels, height, width, sr_depth = gt_imgs.shape
        assert height == width
        
        issr_imgs_1 = issr_imgs.permute(0, 2, 1, 4, 3).reshape(-1, in_channels, sr_depth, width)
        issr_imgs_2 = issr_imgs.permute(0, 3, 1, 2, 4).reshape(-1, in_channels, sr_depth, height)
        issr_imgs_cat = torch.cat([issr_imgs_1, issr_imgs_2])
        
        gt_imgs_1 = gt_imgs.permute(0, 2, 1, 3, 4).reshape(-1, in_channels, sr_depth, width)
        gt_imgs_2 = gt_imgs.permute(0, 3, 1, 2, 4).reshape(-1, in_channels, sr_depth, height,)
        gt_imgs_cat = torch.cat([gt_imgs_1, gt_imgs_2])
    
        num_slices = issr_imgs_cat.size(0)
        selected_slices = np.random.choice(np.arange(num_slices), num_displayed_slices)  

        merged_slices = torch.zeros(size=(num_displayed_slices, in_channels, sr_depth, height * 2 + 10), dtype=gt_imgs.dtype, device=gt_imgs.device)
        merged_slices[:, :, :, :height] = issr_imgs_cat[selected_slices, :, :, :]
        merged_slices[:, :, :, -height:] = gt_imgs_cat[selected_slices, :, :, :]
        
        return merged_slices

    def l1_loss(self, y_sr: torch.Tensor, y: torch.Tensor):
        y_hat = y_sr[:, :, :, :, 2::4]
        return F.l1_loss(y_hat, y)

    ##################################################################################
    ###             adverarial loss and its util functions                         ###
    ##################################################################################
    def adversarial_loss(self, dis_out: torch.Tensor, is_real: bool):
        if is_real:
            return F.binary_cross_entropy_with_logits(dis_out, self.real_label)
        else:
            return F.binary_cross_entropy_with_logits(dis_out, self.fake_label)
        
    def sample_slices(self, y_or_y_sr: torch.Tensor, is_real: bool):
        if is_real:
            y = y_or_y_sr
            batch_size, in_channels, height, width, depth = y.shape
            y = y.permute(0, 4, 1, 2, 3).reshape(-1, in_channels, height, width)
            
            num_slices = y.size(0)
            selected_slices = np.random.choice(np.arange(num_slices), self.num_sample_slices)
            return y[selected_slices, :, :, :]
        else:
            y_sr = y_or_y_sr
            batch_size, in_channels, height, width, sr_depth = y_sr.shape
            y1 = y_sr.permute(0, 2, 1, 3, 4).reshape(-1, in_channels, width, sr_depth)
            y2 = y_sr.permute(0, 3, 1, 2, 4).reshape(-1, in_channels, height, sr_depth)
            y_cat = torch.cat([y1, y2])
            
            num_slices = y_cat.size(0)
            selected_slices = np.random.choice(np.arange(num_slices), self.num_sample_slices)
            
            return y_cat[selected_slices, :, :, :]
    ##################################################################################
    
    @staticmethod
    def initialize_model(model_class, model_params: dict):
        class_args = inspect.getargspec(model_class.__init__).args[1:]
        inkeys = model_params.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = model_params[arg]
        return model_class(**args1)        
        
    def configure_optimizers(self):
        optimizer_params = self.hparams.optimizer_params
        optimizer_params : dict
        
        lr = optimizer_params["lr"]
        b1 = optimizer_params["b1"]
        b2 = optimizer_params["b2"]
        ratio = optimizer_params["ratio"]

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=lr * ratio, betas=(b1, b2))
        return [opt_g, opt_d], []