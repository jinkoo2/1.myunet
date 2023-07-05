from dataset import LocDataset1, NormalizeCT
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader

from helper import append_line, write_line

from os import listdir, mkdir
from os.path import join, isfile, isdir, exists

from helper import get_latest_model_locnet1

print('torch.__version__=', torch.__version__)

############
# parameters
downsample_grid_size = 64

learning_rate = 0.02
num_epochs = 2
batch_size = 8

out_dir = './_train4'

resampled_input_image_size = 64

#########
# device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device = ', device)

# data_train_dir = '/gpfs/scratch/jinkokim/data/train'
# data_valid_dir = '/gpfs/scratch/jinkokim/data/valid'
data_train_dir = '/gpfs/scratch/jinkokim/data/train'
data_valid_dir = '/gpfs/scratch/jinkokim/data/valid'

if not exists(data_train_dir):
    data_train_dir = './data/train'
    data_valid_dir = './data/valid'

print('data_train_dir=', data_train_dir)
print('data_valid_dir=', data_valid_dir)

#data_train_dir = './data3/train'

###########
# dataset
train_ds = LocDataset1(dir=data_train_dir,
                       downsample_grid_size=resampled_input_image_size,
                       transform=NormalizeCT(),
                       sampled_image_out_dir=None)
train_dr = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)

valid_ds = LocDataset1(dir=data_valid_dir,
                       downsample_grid_size=resampled_input_image_size,
                       transform=NormalizeCT(),
                       sampled_image_out_dir=None)
valid_dr = DataLoader(dataset=valid_ds, batch_size=1, shuffle=False)

############
# model
model, epoch0 = get_latest_model_locnet1(
    out_dir=out_dir, input_image_size=resampled_input_image_size)
model = model.to(device)

num_epochs += epoch0
print('num_epochs=', num_epochs)

##############
# print model
W = resampled_input_image_size
x = torch.randn(size=(1, 1, W, W, W), dtype=torch.float32).to(device)
with torch.no_grad():
    out = model(x)
summary = summary(model, (1, W, W, W))

##############
# train
criterion = nn.MSELoss()
criterion_val = nn.L1Loss()  # mean absolute error
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
itr_file_path = f'{out_dir}/itr.csv'
itr_epoch_file_path = f'{out_dir}/itr.epoch.csv'

if not exists(out_dir):
    mkdir(out_dir)

if epoch0 == 0:
    write_line(itr_file_path, f'epoch, i, loss')
    write_line(itr_epoch_file_path,
               f'epoch, validation loss (L2), validation loss (L1), error_600(mm)')

n_total_steps = len(train_dr)/batch_size

#train_itr = iter(train_dr)

for epoch in range(epoch0+1, num_epochs):
    for i, (images, labels) in enumerate(train_dr):
        # for i in range(1):
        #images, labels = train_itr.next()

        # batch
        images = images.to(device)
        labels = labels.to(device)

        print('images.shape', images.shape)
        print('labels.shape', labels.shape)

        # Forward pass
        outputs = model(images)

        print('output.shape', outputs.shape)

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # write
        append_line(itr_file_path, f'{epoch}, {i}, {loss.item():.4f}')

        print(
            f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    # save model
    PATH = f'{out_dir}/model_{epoch}.mdl'
    print('saving model'+PATH)
    torch.save(model.state_dict(), PATH)

    # calc mean loss over the validation dataset
    N_val = len(valid_dr)
    with torch.no_grad():
        sum_loss_L1 = 0.0
        sum_loss_L2 = 0.0
        for i, (images, labels) in enumerate(valid_dr):
            # batch (size = 1 for validation)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_L2 = criterion(outputs, labels)
            loss_L1 = criterion_val(outputs, labels)
            sum_loss_L2 += loss_L2
            sum_loss_L1 += loss_L1
        mean_loss_L2 = sum_loss_L2/N_val
        mean_loss_L1 = sum_loss_L1/N_val

        error_600 = mean_loss_L1 * 600.0  # assuming one side of the image is 600 mm

        append_line(itr_epoch_file_path,
                    f'{epoch}, {mean_loss_L2.item():.4f}, {mean_loss_L1.item():.4f}, {error_600.item():.1f}')
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Mean Validation Loss L2: {mean_loss_L2.item():.4f}')

print('Finished Training')
