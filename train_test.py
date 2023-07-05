from unet import UNet
from dataset import SegDataset1, NormalizeCT
import numpy as np
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.nn.functional as F
from DiceLoss import DiceLoss

from helper import append_line, write_line

from os import listdir, mkdir
from os.path import join, isfile, isdir, exists

from helper import get_latest_model_unet1

############
# parameters
image_sample_width = 64
image_sample_grid_spacing = 2.0
num_samples_per_image = 4

learning_rate = 0.01
num_epochs = 30
batch_size = 2

out_dir = './_train2'

#########
# device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device = ', device)

###########
# dataset
train_ds = SegDataset1(dir='./data/train',
                       grid_size=image_sample_width,
                       grid_spacing=image_sample_grid_spacing,
                       samples_per_image=num_samples_per_image,
                       transform=NormalizeCT())
train_dr = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)

# test_ds = SegDataset1(dir='./data2/test',
#                       grid_size=image_sample_width,
#                       grid_spacing=image_sample_grid_spacing,
#                       samples_per_image=num_samples_per_image,
#                       transform=NormalizeCT())
# test_dr = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

############
# model
model, epoch0 = get_latest_model_unet1(out_dir)
model = model.to(device)

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
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
itr_file_path = f'{out_dir}/itr.csv'

if not exists(out_dir):
    mkdir(out_dir)

if epoch0 == 0:
    write_line(itr_file_path, f'epoch, i, loss')

n_total_steps = len(train_dr)/batch_size

for epoch in range(epoch0+1, num_epochs):
    for i, (images, labels) in enumerate(train_dr):

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

        # if (i+1) % 100 == 0:
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    # save model
    PATH = f'{out_dir}/model_{epoch}.mdl'
    print('saving model'+PATH)
    torch.save(model.state_dict(), PATH)


print('Finished Training')

# PATH = './cnn.pth'
# torch.save(model.state_dict(), PATH)
