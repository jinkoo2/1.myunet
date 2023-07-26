import torch
import torch.nn as nn
from DiceLoss import DiceLoss

def get_loss_function(name):
    if name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'L1Loss':
        return nn.L1Loss()
    elif name == 'DiceLoss':
        return DiceLoss()
    else:
        raise "Invalid loss_func:"+name


def get_optimizer(name, model, learning_rate):
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise "Invalid optimizer:"+name