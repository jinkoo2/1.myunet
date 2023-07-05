import numpy as np
import torch
from torchsummary import summary

from unet import UNet

model = UNet(in_channels=1,
             out_channels=1,
             n_blocks=3,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3)

W = 64
x = torch.randn(size=(1, 1, W, W, W), dtype=torch.float32)
with torch.no_grad():
    out = model(x)
#print(f'Out: {out.shape}')

summary = summary(model, (1, W, W, W))

print('done')
