import numpy as np
import torch
from torchsummary import summary

from locnet import LocNet

W = 64

model = LocNet(in_channels=1,
                out_channels=1,
                n_blocks=2,
                start_filters=8,
                activation='relu',
                normalization=None,
                conv_mode='same',
                input_image_size = W,
                dim=3)

x = torch.randn(size=(4, 1, W, W, W), dtype=torch.float32)
with torch.no_grad():
    out = model(x)
    print(f'Out: {out.shape}')

summary = summary(model, (1, W, W, W))

print('done')
