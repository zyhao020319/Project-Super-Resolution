import torch


def test_BraTs2018_datamodule():
    from experiments.base_configs.materials.lit_BraTs2018_datamodule import config  # 导入训练参数（超分辨率，批次大小，线程数）
    from materials.lit_BraTs2018_datamodule import LitBraTs2018Datamodule
    
    brats2018_datamodule = LitBraTs2018Datamodule(**config)  # 加载数据集
    brats2018_datamodule.setup()
    
    train_dataloader = brats2018_datamodule.train_dataloader()
    print("train dataset : ")
    for batch_moving_images, batch_fixed_images, batch_are_selected in train_dataloader:
        print(batch_moving_images.shape, batch_fixed_images.shape, batch_are_selected.shape)
        print(batch_moving_images.dtype, batch_fixed_images.dtype, batch_are_selected.dtype)
    print()    
    
    val_dataloader = brats2018_datamodule.val_dataloader()
    print("val dataset : ")
    for batch_moving_images, batch_fixed_images, batch_are_selected in val_dataloader:
        print(batch_moving_images.shape, batch_fixed_images.shape, batch_are_selected.shape)
        print(batch_moving_images.dtype, batch_fixed_images.dtype, batch_are_selected.dtype)
    print()    
    
    test_dataloader = brats2018_datamodule.test_dataloader()
    print("test dataset : ")
    for batch_moving_images, batch_fixed_images, batch_are_selected in test_dataloader:
        print(batch_moving_images.shape, batch_fixed_images.shape, batch_are_selected.shape)
        print(batch_moving_images.dtype, batch_fixed_images.dtype, batch_are_selected.dtype)
    print()


def test_VxmDenseFractalUnet():
    from methods.networks.vm_fractalunet import VxmDenseFractalUnet
    
    vxm = VxmDenseFractalUnet(
        inshape=(128, 128),
        nb_unet_features=16,
        nb_unet_levels=4,
        unet_feat_mult=2,
        nb_unet_conv_per_level=2,
        int_downsize=1
    )
    moving_image = torch.randn(1, 4, 128, 128)
    fixed_image = torch.randn(1, 4, 128, 128)
    are_selected = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    y_source, pos_flow = vxm.forward(moving_image, fixed_image, are_selected)
    print(y_source.shape)
    print(pos_flow.shape)    


if __name__ == "__main__":
    test_BraTs2018_datamodule()
