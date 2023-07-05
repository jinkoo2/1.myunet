import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from os import listdir, mkdir
from os.path import join, isfile, isdir
import shutil

from dataset import SegDataset1, NormalizeCT

dataset = SegDataset1(dir='./data/train',
                    grid_size=64,
                    grid_spacing=2.0,
                    samples_per_image=10,
                    transform=NormalizeCT(), 
                    sampled_image_out_dir='./_train4')
print(len(dataset))
ct, rectum = dataset[0]
print('ct.shape=', ct.shape)
print('rectum.shape=', rectum.shape)
print('ct.type=', type(ct))
print('rectum.type=', type(rectum))
print('ct.dtype=', ct.dtype)
print('rectum.dtype=', rectum.dtype)
# with open('ct.npy', 'wb') as f:
#     np.save(f, ct)

# with open('rectum.npy', 'wb') as f:
#     np.save(f, rectum)


# dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False)

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print('features.shape=',features.shape)
# print('labels.shape=',labels.shape)
