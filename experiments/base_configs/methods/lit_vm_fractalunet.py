from easydict import EasyDict


class LitVMFractalUnetConfigs:
    def __init__(self):
        #----------------------------------------------#
        # 形变场可视化的相关参数
        #----------------------------------------------#
        self.vis = EasyDict(
            size         = (240, 240),
            path         = "results/grid.png",
            grid_density = 10,
            batch_size   = 16,
            step         = 100
        )
        #----------------------------------------------#
        # slice interpolation net的参数
        #----------------------------------------------#
        self.bidir = True
        self.int_downsize = 1
    
        self.vxm = EasyDict(
            inshape = (240, 240),
            nb_unet_features=16,
            nb_unet_levels=4,
            unet_feat_mult=2,
            nb_unet_conv_per_level=2,
            int_downsize=1           
        )
        
        #----------------------------------------------#
        # Adam的参数
        #----------------------------------------------#
        self.adam_optimizer = EasyDict(
            lr   = 1e-3,
            betas = (0.5, 0.99)
        )
        #----------------------------------------------#
        # plateau sheduler的参数
        #----------------------------------------------#
        self.cos_anneal_scheduler = EasyDict(
            T_max = 5
        )
        self.lr_warmup = EasyDict(
            num_warmup_steps = 10000
        )
        #----------------------------------------------#
        # 损失函数的权重调整
        #----------------------------------------------#
        self.loss = EasyDict(
            bidir = self.bidir,
            int_downsize = self.int_downsize,
            weight = 0.01
        )
        
        self.test_output_save_dir = ""