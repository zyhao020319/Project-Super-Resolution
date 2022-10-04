import inspect

import torch


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