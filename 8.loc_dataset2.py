import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from image_coord import image_coord

from dataset import LocDataset1, NormalizeCT

dataset = LocDataset1(dir='./data/train',
                      downsample_grid_size=64,
                      transform=NormalizeCT(),
                      sampled_image_out_dir=None)
N = len(dataset)
print('N=', N)

for i in range(1):
    sample, img_coord, bbox_o = dataset.get_item(i)
    ct, bbox_u = sample
    case_name = dataset.image_dirname_list[i]
    print(i)
    print(case_name)
    print(img_coord)
    print(bbox_o)
    print(bbox_u)
 
# print('ct.shape=', ct.shape)
# print('ct.type=', type(ct))
# print('ct.dtype=', ct.dtype)
# print('organ_bbox_u=', organ_bbox_u)
# print('organ_bbox_u.type=', type(organ_bbox_u))
# print('organ_bbox_u.dtype=', organ_bbox_u.dtype)



print('done')
