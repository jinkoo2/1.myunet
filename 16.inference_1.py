# use 64x64x64 model and packing method for inference

from genericpath import exists
from typing import DefaultDict
from unet import UNet
from dataset import SegDataset2, NormalizeCT, sample_image
import numpy as np
import torch
from torch.utils.data import DataLoader
from DiceLoss import DiceLoss
import SimpleITK as sitk

from helper import max, get_grid_list_to_cover_rect, read_key_value_pairs
from os.path import join
from os import mkdir
from rect import rect
from helper import get_latest_model_unet1, s2i, s2f


import math


############
# parameters
image_sample_width = 64
image_sample_grid_spacing = 2.0
num_samples_per_image = 10

out_dir = './_test4'
train_out_dir = './_train3'
#test_data_dir = './data/test'

if not exists(out_dir):
    mkdir(out_dir)

#########
# device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device = ', device)

############
# model
model, epoch0 = get_latest_model_unet1(train_out_dir)
model = model.to(device)
n_blocks=3
grid_size = 64
grid_spacing = 2.0
n_border_pixels = int(np.power(2, n_blocks))
valid_grid_size = 64 - n_border_pixels * 2
valid_grid_size_mm = valid_grid_size * grid_spacing
print('valid_grid_size =', valid_grid_size )
print('n_border_pixels =', n_border_pixels )
print('valid_grid_size_mm =', valid_grid_size_mm )

###############
# image
img_dir = '.\\data1\\00001500'
img_path = join(img_dir, 'CT.mhd')
# dict_img = read_key_value_pairs(img_path)

# img_size = s2i(dict_img['DimSize'].strip().split(' '))
# img_org = s2f(dict_img['Offset'].strip().split(' '))
# img_spacing = s2f(dict_img['ElementSpacing'].strip().split(' '))
# img_coord = image_coord(origin=img_org, size=img_size, spacing=img_spacing)

# print('=========[img info]=============')
# print('img_coord=', img_coord)

#######################################################################################################################
# the loc net should prvide the bounding box (center & size), but here we use the known location in the organ info file
organ_path = join(img_dir, 'Rectum.info')
dict_organ = read_key_value_pairs(organ_path)
bbox_w = s2f(dict_organ['bbox'].split(','))
minx, maxx, miny, maxy, minz, maxz = bbox_w
organ_rect_w = rect(low=[minx, miny, minz], high=[maxx, maxy, maxz])
grid_list = get_grid_list_to_cover_rect(organ_rect_w, grid_size, grid_spacing, n_border_pixels)
print(grid_list)           

###############################
# Segment using the grid list
label_pred_th_list = []
i=0
with torch.no_grad():
    for (grid_coord, grid_org_wrt_grid000_I) in grid_list:
        print('=======================')
        print('i=', i)
        print('grid_coord=', grid_coord)
        print('grid_org_wrt_grid000_I=', grid_org_wrt_grid000_I)
        
        # segment for the grid
        ct_sampled_image_path = join(out_dir, f'CT.sampled.{i}.mhd')
        ct_sampled = sample_image(img_path, grid_coord, 3, -1000 , sitk.sitkLinear, ct_sampled_image_path)
        ct_np = sitk.GetArrayFromImage(ct_sampled).astype('float32')

        # scale image intensity
        factor = 1.0/150.0
        ct_np = 2.0/(1+np.exp(-ct_np*factor))-1.0
        
        # add color channel (1, because it's a gray, 1 color per pixel)
        size = list(ct_np.shape)
        size.insert(0, 1)  # insert 1 to the 0th element (color channel)
        size.insert(0, 1)  # insert 1 to the 0th element (batch channel)
        ct_np.resize(size) # ex) [1,1,64,64,64]

        images = torch.from_numpy(ct_np)

        labels_pred = model(images)

        #save prediction label
        labels_pred = torch.sigmoid(labels_pred)
        label_pred = labels_pred[0][0].numpy() # remove the batch and color dim and convert to numpy array
        print('label_pred.shape', label_pred.shape)

        # threshold & cast
        label_pred_th = np.where(label_pred>=0.5, 1.0, 0.0).astype(np.ubyte)

        label_pred_th_list.append(label_pred_th)        

        # conver to sitkImage to save
        pred_image = sitk.GetImageFromArray(label_pred_th)

        # copy image properties
        pred_image.SetSpacing(ct_sampled.GetSpacing())
        pred_image.SetOrigin(ct_sampled.GetOrigin())
        pred_image.SetDirection(ct_sampled.GetDirection())

        # save image
        pred_image_path = f'{out_dir}/Rectum.sampled.pred.{i}.mhd'
        print('saving pred_image...', pred_image_path)
        sitk.WriteImage(pred_image, pred_image_path, True)  # useCompression:True

        i+=1


##################################################
# combine segments using grid_org_wrt_grid000_I

# find the max shifts in I
max_I = [-1,-1,-1]
for (grid_coord, grid_org_wrt_grid000_I) in grid_list:
    max_I[0] = max(max_I[0], grid_org_wrt_grid000_I[0])
    max_I[1] = max(max_I[1], grid_org_wrt_grid000_I[1])
    max_I[2] = max(max_I[2], grid_org_wrt_grid000_I[2])

max_I = np.array(max_I).astype(int)
print('max_I=', max_I)

combined_label_size = max_I+ [grid_size] * 3
print('combined_label_size=', combined_label_size)

combined_label = np.zeros([combined_label_size[2], combined_label_size[1], combined_label_size[0]])
print('combine_label.shape=', combined_label.shape)

n=0
for (grid_coord, grid_org_wrt_grid000_I) in grid_list:
    print('===========================')
    print('n=', n)
    label_pred_th = label_pred_th_list[n]
   
    print('grid_org_wrt_grid000_I=', grid_org_wrt_grid000_I)
    grid_org_wrt_grid000_I = np.array(grid_org_wrt_grid000_I).astype(int)
    i_org = grid_org_wrt_grid000_I[0]
    j_org = grid_org_wrt_grid000_I[1]
    k_org = grid_org_wrt_grid000_I[2]

    for k in range(label_pred.shape[0]):
        for j in range(label_pred.shape[1]):
            for i in range(label_pred.shape[2]):
                a = combined_label[k_org+k,j_org+j,i_org+i]
                b = label_pred_th[k,j,i]
                combined_label[k_org+k,j_org+j,i_org+i] = a+b
    n+=1

# threshold & cast
combined_label = np.where(combined_label>=0.5, 1.0, 0.0).astype(np.ubyte)

# conver to sitkImage to save
pred_image = sitk.GetImageFromArray(combined_label)

# copy image properties
grid_coord_000,_ = grid_list[0]
pred_image.SetSpacing(grid_coord_000.spacing)
pred_image.SetOrigin(grid_coord_000.origin)
pred_image.SetDirection(grid_coord_000.direction)

# select the largest object
connected_component_image = sitk.ConnectedComponent(pred_image)
print('type(connected_components)=', type(connected_component_image))
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(connected_component_image)

label_of_largest_object = -1
n_pixels_of_largest_objects = 0
for label in stats.GetLabels():
    print('================')
    print('label=', label)
    print('type(label)=', type(label))
    n_pixels = stats.GetNumberOfPixels(label)
    print('N pixels=', n_pixels)

    if n_pixels >n_pixels_of_largest_objects:
        n_pixels_of_largest_objects = n_pixels
        label_of_largest_object = label
print('label_of_largest_object=', label_of_largest_object)    
print('n_pixels_of_largest_objects=', n_pixels_of_largest_objects)    

img_th = sitk.Cast(sitk.Threshold(connected_component_image, label_of_largest_object-0.5, label_of_largest_object+0.5, 0.0), sitk.sitkUInt8)

# save image
pred_image_path = f'{out_dir}/Rectum.sampled.pred.mhd'
print('saving pred_image...', pred_image_path)
sitk.WriteImage(img_th, pred_image_path, True)  # useCompression:True

print('done')


