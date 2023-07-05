from unet import UNet
from dataset import SegDataset2, NormalizeCT
import numpy as np
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.nn.functional as F
from DiceLoss import DiceLoss

from helper import append_line, write_line

from os import listdir, mkdir
from os.path import join, isfile, isdir, exists

from helper import get_latest_model_unet2

print('torch.__version__=', torch.__version__)

############
# parameters
image_sample_width = 128
image_sample_grid_spacing = 2.0

learning_rate = 0.05
num_epochs = 1000
batch_size = 4

out_dir = './_train10'
#########
# device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device = ', device)

data_train_dir = '/gpfs/scratch/jinkokim/data/train'
data_valid_dir = '/gpfs/scratch/jinkokim/data/valid'

if not exists(data_train_dir):
    data_train_dir = './data/train'
    data_valid_dir = './data/valid'

#data_train_dir = './data3/train'

###########
# dataset
train_ds = SegDataset2(dir=data_train_dir,
                       grid_size=image_sample_width,
                       grid_spacing=image_sample_grid_spacing,
                       transform=NormalizeCT())
train_dr = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)

valid_ds = SegDataset2(dir=data_valid_dir,
                       grid_size=image_sample_width,
                       grid_spacing=image_sample_grid_spacing,
                       transform=NormalizeCT())
valid_dr = DataLoader(dataset=valid_ds, batch_size=1, shuffle=False)

############
# model
model, epoch0 = get_latest_model_unet2(out_dir)
model = model.to(device)

num_epochs += epoch0
print('num_epochs=', num_epochs)

##############
# print model
W = image_sample_width
x = torch.randn(size=(1, 1, W, W, W), dtype=torch.float32).to(device)
with torch.no_grad():
    out = model(x)
summary = summary(model, (1, W, W, W))

##############
# train
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
itr_file_path = f'{out_dir}/itr.csv'
itr_epoch_file_path = f'{out_dir}/itr.epoch.csv'

if not exists(out_dir):
    mkdir(out_dir)

if epoch0 == 0:
    write_line(itr_file_path, f'epoch, i, loss')
    write_line(itr_epoch_file_path, f'epoch, loss')

n_total_steps = len(train_dr)/batch_size

train_itr = iter(train_dr)

for epoch in range(epoch0+1, num_epochs):
    for i, (images, labels) in enumerate(train_dr):
        # for i in range(1):
        #     images, labels = train_itr.next()

        # batch
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # print('output.shape', outputs.shape)
        # print('labels.shape', labels.shape)

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
    with torch.no_grad():
        sum_loss = 0.0
        for i, (images, labels) in enumerate(valid_dr):
            # batch (size = 1 for validation)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            sum_loss += loss
        mean_valid_loss = sum_loss/len(valid_dr)
        append_line(itr_epoch_file_path,
                    f'{epoch}, {mean_valid_loss.item():.4f}')
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Mean Validation Loss: {mean_valid_loss.item():.4f}')

print('Finished Training')
