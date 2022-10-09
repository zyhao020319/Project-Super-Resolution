import codecs
import yaml
import os
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from monai.networks.layers.factories import Norm
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from materials import LitISSRLoader
from methods.networks.nets import UNet2d5, resnet50, resnet18
from methods.networks import ISSRGenerator, ISSRDiscriminator
from methods import LitISSRGAN
from experiments.dataset_configs.brats2018slice_configs import STAGE, MODALITY


def test_UNet2d5():
    model = UNet2d5(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 48, 64, 80, 96),
    strides=(
        (2, 2, 1),
        (2, 2, 1),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
    ),
    kernel_sizes=(
        (3, 3, 1),
        (3, 3, 1),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    sample_kernel_sizes=(
        (3, 3, 1),
        (3, 3, 1),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.1,
    # attention_module=False
    ).to("cpu")
    summary(model, input_size=(1, 224, 224, 32))
    # torch.save(model.state_dict(), "results/UNet2d5_spvPA.pt")
    # x = torch.rand(1, 1, 224, 224, 32)
    # y = model(x)
    # loss = y.mean()
    # vis_graph = make_dot(loss, params=dict(model.named_parameters()), show_saved=False)
    # vis_graph.directory = "results"
    # vis_graph.format = "png"
    # vis_graph.view()


def test_resnet18():
    model = resnet18(in_channels=1).to("cpu")
    summary(model, input_size=(1, 128, 128))
    
    
def test_resnet50():
    model = resnet50(in_channels=1).to("cpu")
    summary(model, input_size=(1, 128, 128))
    

def test_gen():
    model = ISSRGenerator(in_channels=1, sample_slices=16, sr_scale=4).to("cpu")
    summary(model, input_size=(1, 128, 128, 32))


def test_dis():
    model = ISSRDiscriminator(in_channels=1).to("cpu")
    summary(model, input_size=(1, 128, 128))


def test_data_loader():
    lit_issr_loader = LitISSRLoader(batch_size=4, num_workers=4, modality="t1")
    lit_issr_loader.setup()
    train_loader = lit_issr_loader.train_dataloader()
    test_loader = lit_issr_loader.val_dataloader()
    
    min_val = torch.inf
    max_val = -torch.inf
    for _, imgs, _ in itertools.chain(train_loader, test_loader):
        min_val = min(min_val, torch.min(imgs).item())
        max_val = max(max_val, torch.max(imgs).item())
        
    print(min_val)
    print(max_val)

def test_init_lit_issr_gan():
    config_file_path = 'experiments/test_program.yaml'
    with codecs.open(config_file_path, 'r', 'utf-8') as file:
        param_groups = yaml.load(file, Loader=yaml.FullLoader)

    tensorboard_params = param_groups["tensorboard"]
    dataset_params = param_groups["dataset"]
    model_params = param_groups["model"]
    optimizer_params = param_groups["optimizer"]
    trainer_params = param_groups["trainer"] 
    loss_params = param_groups["loss"]   
    
    extra_params = dict()
    if trainer_params["gpus"] is not None:
        extra_params["device"] = "cuda:0"
    else:
        extra_params["device"] = "cpu"
    extra_params["dis_out_shape"] = [loss_params["num_sample_slices"]] + loss_params["dis_out_shape"]
    
    extra_params["device"] = "cpu"
    lit_gan_model = LitISSRGAN(model_params, optimizer_params, loss_params, extra_params)


def test_lit_issr_gan():
    config_file_path = "experiments/test_program.yaml"
    with codecs.open(config_file_path, 'r', 'utf-8') as file:
        param_groups = yaml.load(file, Loader=yaml.FullLoader)

    tensorboard_params = param_groups["tensorboard"]
    dataset_params = param_groups["dataset"]
    model_params = param_groups["model"]
    optimizer_params = param_groups["optimizer"]
    trainer_params = param_groups["trainer"] 
    loss_params = param_groups["loss"]   

    os.makedirs(tensorboard_params["save_dir"], exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=tensorboard_params["save_dir"], name=tensorboard_params["exp_name"])
    
    lit_data_module = LitISSRLoader(**dataset_params)
    
    extra_params = dict()
    if trainer_params["gpus"] is not None:
        extra_params["device"] = "cuda:0"
    else:
        extra_params["device"] = "cpu"
    extra_params["dis_out_shape"] = [loss_params["num_sample_slices"]] + loss_params["dis_out_shape"]
    
    lit_gan_model = LitISSRGAN(model_params, optimizer_params, loss_params, extra_params)    

    trainer = pl.Trainer(logger=tb_logger, progress_bar_refresh_rate=0, limit_train_batches=0.1, limit_val_batches=0.1, **trainer_params)
    
    trainer.fit(model=lit_gan_model, datamodule=lit_data_module)  
    
    
def test_dataset_statistical_distribution():
    lit_issr_loader = LitISSRLoader(batch_size=4, num_workers=4, modality="t1")
    lit_issr_loader.setup()
    train_loader = lit_issr_loader.train_dataloader()
    test_loader = lit_issr_loader.val_dataloader()
    
    count = np.zeros(shape=(32768,), dtype=np.int64)
    for batch_idx, imgs in enumerate(itertools.chain(train_loader, test_loader)):
        print(batch_idx)
        imgs : torch.Tensor
        imgs = imgs.numpy().reshape(-1,)
        for pixel in imgs:
            count[pixel] += 1
    
    np.savetxt("results/dataset_voxel_intensity_count.txt", count)
    

def plot_dataset_statistical_distribution():
    import matplotlib.pyplot as plt
    count = np.loadtxt("results/dataset_voxel_intensity_count.txt", dtype=np.float32)
    count = count[1:]
    plt.plot(np.arange(len(count)), count)
    plt.show()


if __name__ == "__main__":
    plot_dataset_statistical_distribution()
    