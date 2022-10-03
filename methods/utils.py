import inspect

import torch
import torch.nn.functional as F


def initialize_class(class_name: object, class_params: dict):
    class_args = inspect.getargspec(class_name.__init__).args[1:]
    inkeys = class_params.keys()
    args1 = {}
    for arg in class_args:
        if arg in inkeys:
            args1[arg] = class_params[arg]
    return class_name(**args1)   


def normalize(pred_slice: torch.Tensor, target_slice: torch.Tensor):
    max_, min_ = torch.max(target_slice), torch.min(target_slice)
    
    image = target_slice.type(torch.float32)
    target_slice = ((image - min_) / (max_- min_) * 255).type(torch.uint8) 
    
    pred_slice = pred_slice.clamp(min_, max_)
    image = pred_slice.type(torch.float32)
    pred_slice = ((image - min_) / (max_- min_) * 255).type(torch.uint8)       
    
    return [pred_slice, target_slice]


def stack_images(image_list : list):
    #------------------------------------------------#
    # image_list : list of image tensors
    # 图像的大小 : [C, H1, W1]
    # 注意不同图像通道数一样，但是可能宽高不一样
    #------------------------------------------------#
    max_height = max(image.shape[1] for image in image_list)
    max_width  = max(image.shape[2] for image in image_list)
    
    canvases = list()
    for image in image_list:
        image : torch.Tensor
        x_diff = (max_height - image.shape[1])
        y_diff = (max_width - image.shape[2])
        canvas = F.pad(image, pad=[y_diff // 2, y_diff - y_diff // 2, x_diff // 2, x_diff - x_diff // 2], mode="constant", value=torch.min(image))
        canvases.append(canvas)
    
    canvases = torch.stack(canvases, dim=0)
    return canvases