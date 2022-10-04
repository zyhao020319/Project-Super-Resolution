import torch
<<<<<<< HEAD
from torchsummary import summary

from experiments.dataset_configs.brats2018slice_configs import BraTs2018SliceConfigs
from materials.lit_BraTs2018_slice_loader import LitBraTs2018SliceLoader
from methods.networks.nets.unet import UNet
from methods.networks.slice_interpolation_net import SliceInterpolationNet


def test_lit_BraTs2018_slice_loader():
    cf = BraTs2018SliceConfigs()
    slice_loader = LitBraTs2018SliceLoader(cf)
    slice_loader.setup()
    print("Train dataset : ", len(slice_loader.train_dataset))
    print("Val dtaset    : ", len(slice_loader.val_dataset))
    print("Test dataset  : ", len(slice_loader.test_dataset))
    
    # train_first_batch = next(iter(slice_loader.train_dataloader()))
    # val_first_batch   = next(iter(slice_loader.val_dataloader()))
    # test_first_batch  = next(iter(slice_loader.test_dataloader()))
    train_data_loader = slice_loader.train_dataloader()
    val_data_loader   = slice_loader.val_dataloader()
    test_data_loader  = slice_loader.test_dataloader()
    for train_batch in train_data_loader:
        print("First train batch : ", train_batch[0].shape, train_batch[1].shape)
        break
    for val_batch in val_data_loader:
        print("First val batch   : ", val_batch[0].shape, val_batch[1].shape)
        break
    for test_batch in test_data_loader:
        print("First test batch  : ", test_batch[0].shape, test_batch[1].shape)
        break


def test_UNet():
    unet = UNet(in_channels=2, out_channels=2, conv_kernel_size=(3, 3), ngf=16, nlayers=4, bilinear=True)
    summary(unet, input_size=(2, 224, 224))
    
    
def test_slice_interpolation_net():
    sin = SliceInterpolationNet(in_channels=2, out_channels=2, ngf=4, input_shape=(224, 224))
    x0 = torch.zeros(size=(1, 1, 224, 224), dtype=torch.float32)
    x1 = torch.zeros(size=(1, 1, 224, 224), dtype=torch.float32)
    ys = sin(x0, x1)
    print(ys[0].shape, ys[1].shape)


def test_voxelmorph_unet():
    from methods.networks.voxelmorph.networks import Unet
    unet2d = Unet(
        inshape=(128, 128),
        infeats=1
    )
    x = torch.zeros(1, 1, 128, 128)
    y = unet2d(x)
    print(y.shape)


def test_voxelmorph():
    from methods.networks.voxelmorph.networks import VxmDense
    vxm = VxmDense(
        inshape=(128, 128),
        int_downsize=1
    )
    moving_image = torch.randn(1, 1, 128, 128)
    fixed_image = torch.randn(1, 1, 128, 128)
    y_source, pos_flow = vxm.forward(moving_image, fixed_image)
    print(y_source.shape)
    print(pos_flow.shape)


if __name__ == "__main__":
    # test_lit_BraTs2018_slice_loader()
    test_UNet()
=======


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
>>>>>>> origin/main
