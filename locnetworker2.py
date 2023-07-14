import torch
from locnet import LocNet
from helper import add_a_dim_to_tensor
from dataset import LocDataset3, NormalizeCT
import numpy as np
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from helper import append_line, write_line
from os import mkdir, remove, write
from os.path import join, exists
from param import param_from_json
import SimpleITK as sitk
from image_coord import image_coord
from dataset import sample_image
from rect import rect
import os
import math
from param import param_from_txt
from helper import s2f


from providers.StructuresDataProvider import StructuresDataProvider
from providers.TrainingJobsDataProvider import TrainingJobsDataProvider

from config import get_config

def str_to_None(str):
    if str.strip() == 'None' or str.strip() == 'none':
        return None
    else:
        return str

def mhd_image_files_exist(mhd):
        if not os.path.exists(mhd):
            print('file not found:'+mhd)
            return False
        zraw = mhd.replace('.mhd', '.zraw')
        if not os.path.exists(zraw):
            print('file not found:'+zraw)
            return False
        info = mhd.replace('.mhd', '.info')
        if not os.path.exists(info):
            print('file not found:'+info)
            return False
        return True

class LocNetWorker2():

    def __init__(self, worker_file):

        self.worker_file = worker_file
        self.worker_dir = os.path.dirname(worker_file)

        print('LocNetWorker2.worker_file=', self.worker_file)
        print('LocNetWorker2.worker_dir=', self.worker_dir)
        
        #########
        # device
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('device = ', self.device)

        self.load_from_worker_file(worker_file)

        self.locnet.to(self.device)

    def load_from_worker_file(self, worker_file):
        print(f'loading [{worker_file}]...')
        param = param_from_json(worker_file)

        model = param['Model']

        print(model)

        in_channels = model['in_channels']
        out_channels = model['out_channels']
        n_blocks = model['n_blocks']
        start_filters = model['start_filters']
        activation = str_to_None(model['activation'])
        normalization = str_to_None(model['normalization'])
        conv_mode = str_to_None(model['conv_mode'])
        input_image_size = model['input_image_size']
        dim = model['dim']
        
        train = param['Train']
        model_file = train['model_file'] if 'model_file' in train else ""

        print('=== LocNet input parameters ====')
        print('in_channels=', in_channels)
        print('out_channels=', out_channels)
        print('n_blocks=', n_blocks)
        print('start_filters=', start_filters)
        print('activation=', activation)
        print('normalization=', normalization)
        print('conv_mode=', conv_mode)
        print('input_image_size=', input_image_size)
        print('dim=', dim)
        print('model_file=', model_file)

        locnet = LocNet(in_channels=in_channels,
                        out_channels=out_channels,
                        n_blocks=n_blocks,
                        start_filters=start_filters,
                        activation=activation,
                        normalization=normalization,
                        conv_mode=conv_mode,
                        input_image_size=input_image_size,
                        dim=dim)

        current_epoch = train['current_epoch'] if 'current_epoch' in train else -1

        if param['Status'] != 'New' and current_epoch >= 0 and model_file is not "" and exists(model_file):
            print(f'loading parameters from {model_file} ...')
            if torch.cuda.is_available():
                print('CUDA is available')
                locnet.load_state_dict(torch.load(model_file))
            else:
                print('CUDA is NOT available, mapping to CPU if needed...')
                locnet.load_state_dict(torch.load(
                    model_file, map_location=torch.device('cpu')))
        else:
            print(f'No model_file is loaded.')

        self.locnet = locnet
        self.worker_file = worker_file
        self.param = param

    def update_status(self, status):
        
        # update memory
        self.param['Status'] = status
        
        # update db
        TrainingJobsDataProvider.update({'_id': self.param['_id'], 'Status': status })
        
        print(f'Status chagned to "{status}"')

    
    def download_dataset(self,dataset_name):
        print(f'--- downloading dataset {dataset_name} ---')
        
        files = []
        for sid in self.param['DataSet'][dataset_name]:
            downloaded_files = StructuresDataProvider.download_files(sid)
            files.append(downloaded_files)

        # check if all files available
        for item in files:
            if not mhd_image_files_exist(item['structure_mhd']):
                raise Exception(f'Input image not found:{item["structure_mhd"]}')
            if not mhd_image_files_exist(item['image_mhd']):
                raise Exception(f'Input image not found:{item["image_mhd"]}')

        return files
    
    def train(self):

        print('torch.__version__=', torch.__version__)

        train_p = self.param['Train']

        ############
        # parameters
        learning_rate = train_p["learning_rate"]
        max_num_of_epochs_per_run = train_p["max_num_of_epochs_per_run"]
        batch_size = train_p["batch_size"]
        out_dir = os.path.join(self.worker_dir, 'train')
        network_input_image_size =  self.param['Model']["input_image_size"]
        optimizer = train_p["optimizer"]

        #########
        # device
        device = self.device

        ###############################
        # download data if not exists
        try:
            train_files = self.download_dataset('TrainSet')
            valid_files = self.download_dataset('ValidSet')
        except Exception as e:
            print('Error - File download failed!')
            return 

        print(f'# of train images = {len(train_files)}')
        print(f'# of valid images = {len(valid_files)}')
      
        ###########
        # dataset
        train_ds = LocDataset3(input_list=train_files,
                               downsample_grid_size=network_input_image_size,
                               transform=NormalizeCT(),
                               sampled_image_out_dir=None)
        train_dlr = DataLoader(
            dataset=train_ds, batch_size=batch_size, shuffle=True)

        valid_ds = LocDataset3(input_list=valid_files,
                               downsample_grid_size=network_input_image_size,
                               transform=NormalizeCT(),
                               sampled_image_out_dir=None)
        valid_dlr = DataLoader(dataset=valid_ds, batch_size=1, shuffle=False)

        ############
        # model
        model = self.locnet
        model = model.to(device)

        if self.param['Status'] == 'New':
            epoch0 = 0
            max_epoch = max_num_of_epochs_per_run
            epoch_list = []
        else:
            epoch_prev_run = train_p["current_epoch"]
            epoch0 = epoch_prev_run + 1
            max_epoch = epoch0 + max_num_of_epochs_per_run
            epoch_list = train_p['epoch_list'] if 'epoch_list' in train_p else []

        print('epoch0=', epoch0)
        print('max_epoch=', max_epoch)

        ##############
        # print model
        W = network_input_image_size
        x = torch.randn(size=(1, 1, W, W, W), dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x)
        summary(model, (1, W, W, W))

        ##############
        # train

        # training loss function
        loss_func_train = None
        if train_p["loss_func_train"] == 'MSELoss':
            loss_func_train = nn.MSELoss()
        else:
            raise "Invalid loss_func_train:"+train_p["loss_func_train"]

        # validation loss function
        loss_func_valid = None
        if train_p["loss_func_valid"] == 'L1Loss':
            loss_func_valid = nn.L1Loss()
        else:
            raise "Invalid loss_func_valid:"+train_p["loss_func_valid"]

        # optimizer
        optimizer = None
        if train_p["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif train_p["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise "Invalid optimizer:"+train_p["optimizer"]

        # training itr log files
        itr_file_path = f'{out_dir}/itr.csv'
        itr_epoch_file_path = f'{out_dir}/itr.epoch.csv'

        print('out_dir=', out_dir)
        print('itr_file_path=', itr_file_path)
        print('itr_epoch_file_path=', itr_epoch_file_path)
        
        # make out dir
        if not exists(out_dir):
            mkdir(out_dir)

        # if first training, add headers
        if epoch0 == 0:
            write_line(itr_file_path, f'epoch, i, loss')
            write_line(itr_epoch_file_path,
                       f'epoch, validation loss (L2), validation loss (L1), error_600(mm)')

        n_steps_per_epoch = math.floor(len(train_ds)/batch_size)+1

        print(f'len(train_dlr)={len(train_dlr)}')
        print(f'batch_size={batch_size}')
        print(f'n_steps_per_epoch (N/batch_size) ={n_steps_per_epoch}')
     

        # update status
        self.update_status("Training.Started")

        for epoch in range(epoch0, max_epoch):
            train_loss_list = []
            loss_for_epoch = {}
            for i, (images, labels) in enumerate(train_dlr):
                # for i in range(1):
                #images, labels = train_itr.next()

                # batch
                images = images.to(device)
                labels = labels.to(device)

                # print('images.shape', images.shape)
                # print('labels.shape', labels.shape)

                # Forward pass
                outputs = model(images)

                # print('output.shape', outputs.shape)

                loss = loss_func_train(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # write
                append_line(itr_file_path, f'{epoch}, {i}, {loss.item():.4f}')

                print(
                    f'Epoch [{epoch}/{max_epoch}], Step [{i}/{n_steps_per_epoch}], Loss: {loss.item():.4f}')
                
                # save train loss 
                train_loss_list.append(loss.item())

            loss_for_epoch['mean_train_loss_for_epoch'] = np.mean(np.array(train_loss_list))

            # calc mean loss over the validation dataset (each epoch save the validation loss)
            N_valid = len(valid_dlr)
            validation_loss = 1000000000000000000.0  # a random large number
            with torch.no_grad():
                sum_loss_L1 = 0.0
                sum_loss_L2 = 0.0
                for i, (images, labels) in enumerate(valid_dlr):
                    # batch (size = 1 for validation)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss_L2 = loss_func_train(outputs, labels)
                    loss_L1 = loss_func_valid(outputs, labels)
                    sum_loss_L2 += loss_L2
                    sum_loss_L1 += loss_L1
                mean_loss_L2 = sum_loss_L2/N_valid
                mean_loss_L1 = sum_loss_L1/N_valid

                error_600 = mean_loss_L1 * 600.0  # assuming one side of the image is 600 mm

                append_line(itr_epoch_file_path,
                            f'{epoch}, {mean_loss_L2.item():.4f}, {mean_loss_L1.item():.4f}, {error_600.item():.1f}')
                print(
                    f'Epoch [{epoch}/{max_epoch}], Mean Validation Loss L2: {mean_loss_L2.item():.4f}, Mean Validation Loss L1: {mean_loss_L1.item():.4f}')

                loss_for_epoch['mean_valid_loss_L1'] = mean_loss_L1.item()
                loss_for_epoch['mean_valid_loss_L2'] = mean_loss_L2.item()
                loss_for_epoch['mean_valid_error_per_600mm'] = error_600.item()

                # validation loss
                validation_loss = mean_loss_L1.item()

            # if the validation is smallest, save the model
            if validation_loss < train_p["min_validation_loss"]:

                # save the model
                model_file = f'{out_dir}/model_{"{:05d}".format(epoch)}.mdl'
                print('saving model'+model_file)
                torch.save(model.state_dict(), model_file)

                # save the current validation loss as minimum
                train_p["min_validation_loss"] = validation_loss
                train_p["selected_model_epoch"] = epoch
                train_p["model_file"] = model_file

            # save the current epoch
            train_p["current_epoch"] = epoch

            # epoch_list
            epoch_list.append({
                'epoch': epoch,
                'loss': loss_for_epoch
            })

            train_p['epoch_list'] = epoch_list

            # update the param file
            print('saving worker file: '+self.worker_file)
            self.param.save_to_json(self.worker_file)

            # update the job in db
            print('updating db...')
            TrainingJobsDataProvider.update(self.param)

            # stop, if there is a stop.txt file
            stop_file = join(out_dir, 'stop.txt')
            if exists(stop_file):
                print('stop.txt found! exiting training.')
                self.update_status("Training.Stopped")
                return 

        self.update_status("Training.Finished")
        print('Finished Training')

    def locate(self, img_path):
        print('img_path=', img_path)

        # image coordinate
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(img_path)
        file_reader.ReadImageInformation()

        # image bounding box
        img_coord = image_coord(origin=file_reader.GetOrigin(), size=file_reader.GetSize(
        ), spacing=file_reader.GetSpacing(), direction=file_reader.GetDirection())
        print('img_coord=', img_coord)
        print('img_coord.rect_o=', img_coord.rect_o())
        print('img_coord.rect_o.size()=', img_coord.rect_o().size())

        W = self.param['Model']["input_image_size"]
        # downsample the image to downsample_grid_size (ex, 64, 64, 64),
        sample_grid_coord = image_coord(
            origin=img_coord.origin,
            size=[W] * 3,
            spacing=img_coord.rect_o().size(
            )/np.array([W] * 3),
            direction=img_coord.direction
        )

        ct_sampled_image_path = img_path+".sampled.mhd"
        ct_sampled = sample_image(
            img_path, sample_grid_coord, 0, -1000, sitk.sitkLinear, ct_sampled_image_path)

       # sitImage to Numpy
        ct_np = sitk.GetArrayFromImage(ct_sampled).astype('float32')

        # the coefficient of 1/150 will map the range of -300 to 300 to good portion of the outputs;
        # outputs will be in [-1.0, 1.0]
        factor = 1.0/150.0
        ct_np = 2.0/(1+np.exp(-ct_np*factor))-1.0

        images = torch.from_numpy(ct_np)
        add_a_dim_to_tensor(images)
        add_a_dim_to_tensor(images)

        images = images.to(self.device)

        with torch.no_grad():
            labels_pred = self.locnet(images)

            # output in u (center & size)
            output = labels_pred[0].numpy()
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

            # rect in u
            rect_u = rect(low=pred_center_u-pred_size_u/2.0,
                          high=pred_center_u+pred_size_u/2.0)

            # rect in o
            rect_o = rect(low=pred_center_o-pred_size_o/2.0,
                          high=pred_center_o+pred_size_o/2.0)

            # rect in w
            rect_w = rect(low=rect_o.low+img_coord.origin,
                          high=rect_o.high+img_coord.origin)

            print('rect_w=', rect_w)
            print('rect_w.size()=', rect_w.size())

        return rect_w, rect_o, rect_u

    def test(self):

        print('download test dataset')
        try:
            test_files = self.download_dataset('TestSet')
        except Exception as e:
            print('Error - File download failed!')
            return 
        
        print(f'Test dataset N={len(test_files)}')

        print(test_files[0])

        err_mm_list = []
        for item in test_files:
            img_file = item['image_mhd']
            str_file = item['structure_mhd']

            rect_w_pred,_,_ = self.locate(img_file)
            #print('rect_w_pred=',rect_w_pred)

            # the true values
            str_info = str_file.replace('.mhd', '.info')
            bbox = s2f(param_from_txt(str_info)['bbox'].split(','))
            rect_w_true = rect(low=[bbox[0], bbox[2], bbox[4]],
                            high=[bbox[1], bbox[3], bbox[5]])
            #print('rect_w_true=', rect_w_true)

            rect_diff = rect(low=rect_w_pred.low - rect_w_true.low,
                            high=rect_w_pred.high - rect_w_true.high)
            print('rect_diff=', rect_diff)
            err_mm = (np.linalg.norm(rect_diff.low)+np.linalg.norm(rect_diff.low))/2.0

            print(f'error(mm)={err_mm})')
            err_mm_list.append(err_mm)
        
        err_mean = np.mean(np.array(err_mm_list))

        print(f'err_mean(mm)={err_mean}')
        


