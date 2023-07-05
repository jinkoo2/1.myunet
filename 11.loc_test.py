from dataset import LocDataset1, NormalizeCT
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader

from helper import append_line, write_line

from os import listdir, mkdir
from os.path import join, isfile, isdir, exists

from helper import get_latest_model_locnet1, add_a_dim_to_tensor, array_to_string as a2s

print('torch.__version__=', torch.__version__)

############
# parameters
downsample_grid_size = 64
batch_size = 1
out_dir = './_train4'

resampled_input_image_size = 64

#########
# device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device = ', device)

data_test_dir = '/gpfs/scratch/jinkokim/data/test'
if not exists(data_test_dir):
    data_test_dir = './data/test'

print('data_test_dir=', data_test_dir)

itr_test_file_path = join(out_dir, 'itr.test.csv')

###########
# dataset
test_ds = LocDataset1(dir=data_test_dir,
                      downsample_grid_size=resampled_input_image_size,
                      transform=NormalizeCT(),
                      sampled_image_out_dir=None)
#test_dr = DataLoader(dataset=test_ds, batch_size=1, shuffle=False)

############
# model
model, epoch0 = get_latest_model_locnet1(
    out_dir=out_dir, input_image_size=resampled_input_image_size)
model = model.to(device)

##############
# print model
W = resampled_input_image_size
x = torch.randn(size=(1, 1, W, W, W), dtype=torch.float32).to(device)
with torch.no_grad():
    out = model(x)
summary = summary(model, (1, W, W, W))

##############
# train
criterion_L2 = nn.MSELoss()
criterion_L1 = nn.L1Loss()  # mean absolute error

if not exists(out_dir):
    mkdir(out_dir)


write_line(itr_test_file_path,
           'i, case_name, loss_L2, loss_L1, label.0,1,2,3,4,5, pred.0,1,2,3,4,5, error_u.0,1,2,3,4,5, error_o.0,1,2,3,4,5')

# calc mean loss over the validation dataset
N_test = len(test_ds)
print('N_test=', N_test)
with torch.no_grad():
    sum_loss_L1 = 0.0
    sum_loss_L2 = 0.0
    for i in range(N_test):
        print('i=', i)
        sample, img_coord, bbox_o = test_ds.get_item(i)
        image, label = sample
        case_name = test_ds.image_dirname_list[i]
        print('i=', i)
        print('case_name=', case_name)
        print('img_coord=', img_coord)
        print('bbox_o=', bbox_o)

        organ_center_o = bbox_o.center()
        organ_size_o = bbox_o.size()
        print('organ_center_o=', organ_center_o)
        print('organ_size_o=', organ_size_o)

        # add a batch dim
        images = image
        labels = label
        add_a_dim_to_tensor(images)
        add_a_dim_to_tensor(labels)

        # batch (size = 1 for validation)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss_L2 = criterion_L2(outputs, labels)
        loss_L1 = criterion_L1(outputs, labels)
        sum_loss_L2 += loss_L2
        sum_loss_L1 += loss_L1

        # output in u (center & size)
        output = outputs[0].numpy()
        pred_center_u = np.array([output[0], output[1], output[2]])
        pred_size_u = np.array([output[3], output[4], output[5]])
        print('output=', output)
        print('pred_center_u=', pred_center_u)
        print('pred_size_u=', pred_size_u)

        # convert to o
        img_size_o = img_coord.rect_o().size()
        pred_center_o = pred_center_u * img_size_o
        pred_size_o = pred_size_u * img_size_o

        print('img_size_o=', img_size_o)
        print('pred_center_o=', pred_center_o)
        print('pred_size_o=', pred_size_o)

        error_u = outputs[0].numpy() - labels[0].numpy()
        print('error_u=', error_u)
        sizesize = np.array(img_size_o.tolist() * 2)  # [w,h,z,w,h,z]
        print('sizesize=', sizesize)
        error_o = error_u * sizesize
        print('error_o=', error_o)

        append_line(itr_test_file_path,
                    f'{i}, {case_name}, {loss_L2.item():.4f}, {loss_L1.item():.4f}, {a2s(labels[0].numpy())}, {a2s(outputs[0].numpy())},{a2s(error_u)},{a2s(error_o)}')

    mean_loss_L2 = sum_loss_L2/N_test
    mean_loss_L1 = sum_loss_L1/N_test


print('Finished Testing')
