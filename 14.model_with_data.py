from dataset import SegDataset2, NormalizeCT
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from unet import UNet


#########
# device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device = ', device)


model = UNet(in_channels=1,
             out_channels=1,
             n_blocks=4,
             start_filters=8,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3).to(device)
W = 128

# x = torch.randn(size=(1, 1, W, W, W), dtype=torch.float32)
# with torch.no_grad():
#     out = model(x)
# #print(f'Out: {out.shape}')
summary = summary(model, (1, W, W, W))

#######################
# test model with data
print('test with a random data')
x = torch.randn(size=(1, 1, W, W, W), dtype=torch.float32).to(device)
with torch.no_grad():
    out = model(x)
    print(f'Out: {out.shape}')

##########################
# test model with dataset

dataset = SegDataset2(dir='./data/train',
                      grid_size=W,
                      grid_spacing=2.0,
                      transform=NormalizeCT())
# print(len(dataset))
# ct, rectum = dataset[49]
# print('ct.shape=', ct.shape)
# print('rectum.shape=', rectum.shape)
# print('ct.type=', type(ct))
# print('rectum.type=', type(rectum))
# print('ct.dtype=', ct.dtype)
# print('rectum.dtype=', rectum.dtype)
# with open('ct.npy', 'wb') as f:
#     np.save(f, ct)

# with open('rectum.npy', 'wb') as f:
#     np.save(f, rectum)


dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print('features.shape=', features.shape)
print('labels.shape=', labels.shape)
print('type(features)=', type(features))
print('type(labels)=', type(labels))

#model = model.to(device)
with torch.no_grad():
    features = features.to(device)
    out = model(features)
    print(f'Out: {out.shape}')

#############
print('done')
