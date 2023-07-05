# make a training reqest
from param import Param

out_param_file = 'c:/data/_db/_tr_reqs/locnet.bladder.train.json'
print('out_param_file=', out_param_file)

p = Param()

p['name'] = 'locnet_bladder_train1'
p['dataset'] = 'cont.ds.bladder.global_2021.09.15.json'
p['locnet'] = {
    'input_image_size': 64,
    'in_channels': 1,
    'out_channels': 1,
    'n_blocks': 2,
    'start_filters': 32,
    'activation': 'relu',
    'normalization': 'batch',
    'conv_mode': 'same',
    'dim': 3
}
p['train'] = {
    'optim':{
        'type': 'SGD',
        'learning_rate': 0.02
    },
    'loss_fn_train': 'MSELoss',
    'loss_fn_valid1': 'MSELoss',
    'loss_fn_valid2': 'L1Loss',
    'num_epochs': 1000,
    'batch_size': 8,
}

p.save_to_json(out_param_file)

print('done.')
