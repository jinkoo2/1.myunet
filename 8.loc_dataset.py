import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from image_coord import image_coord

from dataset import LocDataset1, NormalizeCT

dataset = LocDataset1(dir='./data/train',
                      downsample_grid_size=64,
                      transform=NormalizeCT(),
                      sampled_image_out_dir='./_train4')
print(len(dataset))
sample = dataset[0]
ct, organ_bbox_u = sample

print('ct.shape=', ct.shape)
print('ct.type=', type(ct))
print('ct.dtype=', ct.dtype)
print('organ_bbox_u=', organ_bbox_u)
print('organ_bbox_u.type=', type(organ_bbox_u))
print('organ_bbox_u.dtype=', organ_bbox_u.dtype)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False)

dataiter = iter(dataloader)
data = next(dataiter)
features, labels = data

print('features.shape=',features.shape)
print('labels.shape=',labels.shape)

print('done')
