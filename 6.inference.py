
from unet import UNet
from dataset import SegDataset1, NormalizeCT
import numpy as np
import torch
from torch.utils.data import DataLoader
from DiceLoss import DiceLoss
import SimpleITK as sitk

from helper import append_line, write_line, add_a_dim_to_tensor
from os.path import join, exists
from os import mkdir


from helper import get_latest_model_unet1

############
# parameters
image_sample_width = 64
image_sample_grid_spacing = 2.0
num_samples_per_image = 10

out_dir = './_test3'
train_out_dir = './_train3'
test_data_dir = './data_test/test'

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

###############
# data to test
test_ds = SegDataset1(dir=test_data_dir,
                      grid_size=image_sample_width,
                      grid_spacing=image_sample_grid_spacing,
                      samples_per_image=num_samples_per_image,
                      transform=NormalizeCT(),
                      sampled_image_out_dir=out_dir)

#test_ds[0]

test_dr = DataLoader(dataset=test_ds, batch_size=1, shuffle=False)

print('N=', len(test_ds))

criterion = DiceLoss()

dice_list = []

file_dice = join(out_dir, 'test_dice.csv')
write_line(file_dice, f'n,dice')

with torch.no_grad():
    for i in range(len(test_ds)):
    #for i in range(1):
        print(f'i={i}')
        image, label = test_ds[i]
        add_a_dim_to_tensor(image)
        add_a_dim_to_tensor(label)
        label_pred = model(image)
        loss = criterion(label_pred, label)

        print(f'loss: {loss.item()}')

        dice = 1.0 - loss
        line = f'{i},{dice}'
        append_line(file_dice, line)
        dice_list.append(dice)

        #save prediction label
        label_pred = torch.sigmoid(label_pred)
        label_pred = label_pred[0][0].numpy() # remove the batch and color dim and convert to numpy array
        print('label_pres.shape', label_pred.shape)

        # threshold & cast
        label_pred = np.where(label_pred>=0.5, 1.0, 0.0).astype(np.ubyte)
        
        # conver to sitkImage to save
        pred_image = sitk.GetImageFromArray(label_pred)

        # copy image properties
        ref_image_path = f'{out_dir}/CT_sample_{i}.mhd'
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(ref_image_path)
        file_reader.ReadImageInformation()

        pred_image.SetSpacing(file_reader.GetSpacing())
        pred_image.SetOrigin(file_reader.GetOrigin())
        pred_image.SetDirection(file_reader.GetDirection())

        # save image
        pred_image_path = f'{out_dir}/Rectum_sample_{i}_pred.mhd'
        print('saving pred_image...', pred_image_path)
        sitk.WriteImage(pred_image, pred_image_path, True)  # useCompression:True

print('done')
