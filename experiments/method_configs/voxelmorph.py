from easydict import EasyDict

class VoxelMorphConfigs:
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
        self.bidir = False
        self.int_downsize = 1
        
        self.enc_nf = [32, 64, 64, 64]
        self.dec_nf = [64, 64, 64, 64, 64, 32, 32]
        self.vxm = EasyDict(
            inshape = (240, 240),
            nb_unet_features=[self.enc_nf, self.dec_nf],
            bidir=self.bidir,
            int_steps=7,
            int_downsize=self.int_downsize,
            src_feats=4,
            trg_feats=4,            
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
        