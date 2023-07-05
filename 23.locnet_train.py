from abc import abstractproperty
from dataset import LocDataset2, NormalizeCT
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader

from helper import append_line, write_line

from os import listdir, makedirs, mkdir
from os.path import join, isfile, isdir, exists

from helper import get_latest_model_locnet3

from param import param_from_json
from locnet import LocNet

# input training param file
train_param_file = 'C:\\data\\_db\\_tr_reqs\\locnet.bladder.train.json'
local_data_dir = './local_data'

print('torch.__version__=', torch.__version__)

print('train_param_file=', train_param_file)
print('local_data_dir=', local_data_dir)

p = param_from_json(train_param_file)
print('p=', p)


############
# parameters
downsample_grid_size = p['locnet']['input_image_size']
learning_rate = p['train']['learning_rate']
num_epochs = p['train']['num_epochs']
batch_size = p['train']['batch_size']
out_dir = f'{local_data_dir}/_trains/{p["name"]}'
resampled_input_image_size = downsample_grid_size
dataset_file = p['dataset']['file']

print('downsample_grid_size=', downsample_grid_size)
print('learning_rate=', learning_rate)
print('num_epochs=', num_epochs)
print('batch_size=', batch_size)
print('resampled_input_image_size=', resampled_input_image_size)
print('dataset_file=', dataset_file)


if not exists(out_dir):
    makedirs(out_dir)
#########
# device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device = ', device)

# data_train_dir = '/gpfs/scratch/jinkokim/data/train'
# data_valid_dir = '/gpfs/scratch/jinkokim/data/valid'

# if not exists(data_train_dir):
#     data_train_dir = './data/train'
#     data_valid_dir = './data/valid'

# print('data_train_dir=', data_train_dir)
# print('data_valid_dir=', data_valid_dir)

#data_train_dir = './data3/train'


###########
# dataset
train_ds = LocDataset2(dataset_json_file=dataset_file,
                       data_for='train',
                       downsample_grid_size=resampled_input_image_size,
                       transform=NormalizeCT(),
                       sampled_image_out_dir='c:/tmp')

train_dr = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)

valid_ds = LocDataset2(dataset_json_file=dataset_file,
                       data_for='valid',
                       downsample_grid_size=resampled_input_image_size,
                       transform=NormalizeCT(),
                       sampled_image_out_dir=None)
valid_dr = DataLoader(dataset=valid_ds, batch_size=1, shuffle=False)

############
# model
p_locnet = p['locnet']
print(p_locnet)

model = LocNet(in_channels=p_locnet['in_channels'],
               out_channels=p_locnet['out_channels'],
               n_blocks=p_locnet['n_blocks'],
               start_filters=p_locnet['start_filters'],
               activation=p_locnet['activation'],
               normalization=p_locnet['normalization'],
               conv_mode=p_locnet['conv_mode'],
               input_image_size=p_locnet['input_image_size'],
               dim=p_locnet['dim'])

if not exists(out_dir):
    epoch0 = 0

# search previous trained model file
model_files = [fname for fname in listdir(
    out_dir) if fname.startswith('model_') and fname.endswith('.mdl')]
if len(model_files) > 0:
    model_epochs = [int(fname.split('.')[0].split('_')[1])
                    for fname in model_files]
    max_epoch = max(model_epochs)
    latest_model_fname = join(out_dir, f'model_{max_epoch}.mdl')
    print('loading model from... ', latest_model_fname)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(latest_model_fname))
    else:
        model.load_state_dict(torch.load(
            latest_model_fname, map_location=torch.device('cpu')))

    epoch0 = max_epoch
else:
    epoch0 = 0
    print('no model file found. ')

# model, epoch0 = get_latest_model_locnet3(
#     out_dir=out_dir, input_image_size=resampled_input_image_size)

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

def get_loss_fn(type):
    if type == 'MSELoss':
        return nn.MSELoss()
    if type == 'L1Loss':
        return nn.L1Loss()
    if type == 'DiceLoss':
        return DiceLoss()
    raise Exception(f'Invalid loss function type: {type}')


def get_optim(model, param):
    type = param['type']
    if type == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=param['learning_rate'])

    if type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=param['learning_rate'])

    raise Exception(f'Invalid optimizer type: {type}')


##############
# train
loss_fn_train = get_loss_fn(p['train']['loss_fn_train'])
loss_fn_valid1 = get_loss_fn(p['train']['loss_fn_valid1'])
loss_fn_valid2 = get_loss_fn(p['train']['loss_fn_valid1'])
optimizer = get_optim(p['train']['optim'])

itr_file_path = f'{out_dir}/itr.csv'
itr_epoch_file_path = f'{out_dir}/itr.epoch.csv'

if epoch0 == 0:
    write_line(itr_file_path, f'epoch, i, train.loss')
    write_line(itr_epoch_file_path,
               f'epoch, train.loss., validation loss (L1)')

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

        loss = loss_fn_train(outputs, labels)

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
        sum_loss1 = 0.0
        sum_loss2 = 0.0
        for i, (images, labels) in enumerate(valid_dr):
            # batch (size = 1 for validation)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss1 = loss_fn_valid1(outputs, labels)
            loss2 = loss_fn_valid2(outputs, labels)
            sum_loss1 += loss1
            sum_loss2 += loss2
        mean_loss1 = sum_loss1/N_val
        mean_loss2 = sum_loss2/N_val

        append_line(itr_epoch_file_path,
                    f'{epoch}, {mean_loss1.item():.4f}, {mean_loss2.item():.4f}')
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Mean Validation Loss 1: {mean_loss1.item():.4f}, Mean Validation Loss 2: {mean_loss2.item():.4f}')

print('Finished Training')
