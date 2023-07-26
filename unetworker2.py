import torch
from unet import UNet
from helper import add_a_dim_to_tensor, split_directory_path, zip_folder
from dataset import SegDataset3, NormalizeCT
import numpy as np
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from helper import append_line, write_line
from image_helper import get_grid_list_to_cover_rect, mhd_image_files_exist
from os import mkdir, remove, write
from os.path import join, exists
from param import param_from_json, Param
import SimpleITK as sitk
from image_coord import image_coord
from dataset import sample_image
from rect import rect
from DiceLoss import DiceLoss
import os
import math

from providers.StructuresDataProvider import StructuresDataProvider
from providers.TrainingJobsDataProvider import TrainingJobsDataProvider

from train_helper import get_loss_function, get_optimizer

def str_to_None(str):
    if str.strip() == 'None' or str.strip() == 'none':
        return None
    else:
        return str





class UNetWorker2():

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

        self.unet.to(self.device)

    def load_from_worker_file(self, worker_file):
        
        print(f'loading [{worker_file}]...')
        
        p = param_from_json(worker_file)

        # Model        
        model = p['Model']
        in_channels = model['in_channels']
        out_channels = model['out_channels']
        n_blocks = model['n_blocks']
        start_filters = model['start_filters']
        activation = str_to_None(model['activation'])
        normalization = str_to_None(model['normalization'])
        conv_mode = str_to_None(model['conv_mode'])
        input_image_size = model['input_image_size']
        dim = model['dim']

        up_mode = model['up_mode']
        final_activation = str_to_None(model['final_activation'])
        #model_file = p['model_file']

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
        print('up_mode=', up_mode)
        print('final_activation=', final_activation)

        train = p['Train']
        model_file = train['model_file'] if 'model_file' in train else ""

        unet = UNet(in_channels=in_channels,
                    out_channels=out_channels,
                    n_blocks=n_blocks,
                    start_filters=start_filters,
                    activation=activation,
                    normalization=normalization,
                    conv_mode=conv_mode,
                    up_mode=up_mode,
                    final_activation=final_activation,
                    dim=dim)
        
        current_epoch = train['current_epoch'] if 'current_epoch' in train else -1

        if p['Status'] != 'New' and current_epoch >= 0 and model_file is not "" and exists(model_file):
            print(f'loading parameters from {model_file} ...')
            if torch.cuda.is_available():
                print('CUDA is available')
                unet.load_state_dict(torch.load(model_file))
            else:
                print('CUDA is NOT available, mapping to CPU if needed...')
                unet.load_state_dict(torch.load(
                    model_file, map_location=torch.device('cpu')))
        else:
            print(f'No model_file is availble.')

        self.unet = unet
        self.worker_file = worker_file
        self.param = p

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
    def update_status(self, status):
            
            # update memory
            self.param['Status'] = status
            
            # update db
            TrainingJobsDataProvider.update({'_id': self.param['_id'], 'Status': status })
            
            print(f'Status chagned to "{status}"')

    def zip_and_upload_tboard_logdir(self):
        out_dir = os.path.join(self.worker_dir, 'train')
        src = os.path.join(out_dir,'tboard')
        dst = os.path.join(out_dir,'tboard.zip')
        zip_folder(src, dst)
        TrainingJobsDataProvider.upload_file(self.param['_id'], dst)
        
    def train(self):

        print('torch.__version__=', torch.__version__)

        input_image_size = self.param['Model']["input_image_size"]
        input_image_spacing = self.param['Model']["input_image_spacing"]

        train_p = self.param["Train"]

        ############
        # parameters
        learning_rate = train_p["learning_rate"]
        max_num_of_epochs_per_run = train_p["max_num_of_epochs_per_run"]
        samples_per_image = train_p['samples_per_image']
        batch_size = train_p["batch_size"]
        out_dir = os.path.join(self.worker_dir, 'train')
        optimizer = train_p["optimizer"]

        ################
        # tensorboard
        tensorboard = True
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            # default `log_dir` is "runs" - we'll be more specific here
            tb_writer = SummaryWriter(os.path.join(out_dir,'tboard'))

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
        train_ds = SegDataset3(input_list=train_files,
                               grid_size=input_image_size,
                               grid_spacing=input_image_spacing,
                               samples_per_image=samples_per_image,
                               transform=NormalizeCT())
        train_dlr = DataLoader(
            dataset=train_ds, batch_size=batch_size, shuffle=True)

        valid_ds = SegDataset3(input_list=valid_files,
                               grid_size=input_image_size,
                               grid_spacing=input_image_spacing,
                               samples_per_image=samples_per_image,
                               transform=NormalizeCT())
        valid_dlr = DataLoader(dataset=valid_ds, batch_size=1, shuffle=False)

        ############
        # model
        model = self.unet
        model = model.to(device)

        if self.param['Status'] == 'New':
            epoch0 = 0
            max_epoch = max_num_of_epochs_per_run
            epoch_list = []
            train_p['min_validation_loss'] = 100000000000000000000000.0
        else:
            epoch_prev_run = train_p["current_epoch"]
            epoch0 = epoch_prev_run + 1
            max_epoch = epoch0 + max_num_of_epochs_per_run
            epoch_list = train_p['epoch_list'] if 'epoch_list' in train_p else []

        print('epoch0=', epoch0)
        print('max_epoch=', max_epoch)

        ##############
        # print model
        W = input_image_size
        x = torch.randn(size=(1, 1, W, W, W), dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x)
        summary(model, (1, W, W, W))

        # ##################################
        # # add model graph to tensorboard
        # if tensorboard:
        #     tb_writer.add_graph(model, torch.zeros(1, W, W, W))
        #     self.zip_and_upload_tboard_logdir()

        ##############
        # train

        # loss functions
        loss_func_train = get_loss_function(train_p["loss_func_train"])
        loss_func_valid = get_loss_function(train_p["loss_func_valid"])

        # optimizer
        optimizer = get_optimizer(train_p["optimizer"], model, learning_rate)

        # training itr log files
        itr_file_path = os.path.join(out_dir,'itr.csv')
        itr_epoch_file_path = os.path.join(out_dir,'itr.epoch.csv')

        # make out dir
        if not exists(out_dir):
            mkdir(out_dir)

        # if first training, add headers
        if epoch0 == 0:
            write_line(itr_file_path, f'epoch, i, loss')
            write_line(itr_epoch_file_path, f'epoch, validation loss')

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

            loss_for_epoch['mean_train_loss'] = np.mean(np.array(train_loss_list))

            # calc mean loss over the validation dataset (each epoch save the validation loss)
            N_valid = len(valid_dlr)
            validation_loss = 1000000000000000000.0  # a random large number
            with torch.no_grad():
                sum_loss_valid = 0.0
                for i, (images, labels) in enumerate(valid_dlr):
                    # batch (size = 1 for validation)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss_valid = loss_func_valid(outputs, labels)
                    sum_loss_valid += loss_valid
                mean_loss_valid = sum_loss_valid/N_valid
                append_line(itr_epoch_file_path,
                            f'{epoch}, {mean_loss_valid.item():.4f}')
                print(
                    f'Epoch [{epoch}/{max_epoch}], Mean Validation Loss: {mean_loss_valid.item():.4f}')

                loss_for_epoch['mean_valid_loss'] = mean_loss_valid.item()

                # validation loss
                validation_loss = mean_loss_valid.item()

            # if the validation is smallest, save the model
            if validation_loss < train_p["min_validation_loss"]:

                # save the model
                #model_file = f'{out_dir}/model_{"{:05d}".format(epoch)}.mdl'
                model_file = f'{out_dir}/model.mdl'
                print('saving model'+model_file)
                torch.save(model.state_dict(), model_file)

                # save the current validation loss as minimum
                train_p["min_validation_loss"] = validation_loss
                train_p["selected_model_epoch"] = epoch
                train_p["model_file"] = model_file

                # update the model_file
                TrainingJobsDataProvider.upload_file(self.param['_id'], model_file)

            # save the current epoch
            train_p["current_epoch"] = epoch

            # epoch_list
            epoch_list.append({
                'epoch': epoch,
                'loss': loss_for_epoch
            })

            # tensorboard
            if tensorboard:
                tb_writer.add_scalar('train_loss', loss_for_epoch['mean_train_loss'], epoch)
                tb_writer.add_scalar('valid_loss', loss_for_epoch['mean_valid_loss'], epoch)
                self.zip_and_upload_tboard_logdir()

            train_p['epoch_list'] = epoch_list

            # update the param file
            print('saving worker file: '+self.worker_file)
            self.param.save_to_json(self.worker_file)

            # update the job in db
            print('updating db...')
            TrainingJobsDataProvider.update(self.param)

            # upload the itr files
            TrainingJobsDataProvider.upload_file(self.param['_id'], itr_epoch_file_path)
            TrainingJobsDataProvider.upload_file(self.param['_id'], itr_file_path)

            # stop, if there is a stop.txt file
            stop_file = join(out_dir, 'stop.txt')
            if exists(stop_file):
                print('stop.txt found! exiting training.')
                self.update_status("Training.Stopped")
                return

        self.update_status("Training.Finished")
        print('Finished Training')

    def segment(self, img_file, roi_rect_w=None, out_file=None, out_dir_for_debug=None):
        print('========segment()=========')
        print('img_file=', img_file)
        print("roi_rect_w=", roi_rect_w)
        print("out_file=", out_file)

        if roi_rect_w == None:
            print("VERYFY THIS CODE!!!")
            # image file reader
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(img_file)   
            # read the image information without reading the bulk data, compute ROI start and size and read it.
            file_reader.ReadImageInformation()
            img_coord = image_coord(size=file_reader.GetSize(), origin=file_reader.GetOrigin(), spacing=file_reader.GetSpacing())
            roi_rect_w = rect(low=img_coord.origin, high=img_coord.origin+img_coord.size*img_coord.spacing)
            print('roi_rect_w=', roi_rect_w) 

        ###################################################
        # get the list of sampling grid list for roi_rect_w
        print('get the list of sampling grid list for roi_rect_w')
        grid_size = self.param['Model']["input_image_size"]
        grid_spacing = self.param['Model']["input_image_spacing"]
        n_border_pixels = int(np.power(2, self.param['Model']["n_blocks"]))
        grid_list = get_grid_list_to_cover_rect(
            roi_rect_w, grid_size, grid_spacing, n_border_pixels)
        print('grid_list=', grid_list)

        ###############################
        # Segment using the grid list
        label_pred_th_list = []
        model = self.unet
        image_resample_background_pixel_value = self.param['Model']["image_resample_background_pixel_value"]
        i = 0
        with torch.no_grad():
            for (grid_coord, grid_org_wrt_grid000_I) in grid_list:
                print('======= pred for grid =============')
                print('i=', i)
                print('grid_coord=', grid_coord)
                print('grid_org_wrt_grid000_I=', grid_org_wrt_grid000_I)

                # segment for the grid
                img_sampled_image_path = join(
                    out_dir_for_debug, f'img.sampled.{i}.mhd') if out_dir_for_debug is not None else None
                img_sampled = sample_image(
                    img_file, grid_coord, 3, image_resample_background_pixel_value, sitk.sitkLinear, img_sampled_image_path)
                img_np = sitk.GetArrayFromImage(img_sampled).astype('float32')

                # scale image intensity
                factor = 1.0/150.0
                img_np = 2.0/(1+np.exp(-img_np*factor))-1.0

                # add color channel (1, because it's a gray, 1 color per pixel)
                size = list(img_np.shape)
                # insert 1 to the 0th element (color channel)
                size.insert(0, 1)
                # insert 1 to the 0th element (batch channel)
                size.insert(0, 1)
                img_np.resize(size)  # ex) [1,1,64,64,64]

                images = torch.from_numpy(img_np)

                labels_pred = model(images)

                # save prediction label
                labels_pred = torch.sigmoid(labels_pred)
                # remove the batch and color dim and convert to numpy array
                label_pred = labels_pred[0][0].numpy()
                print('label_pred.shape', label_pred.shape)

                # threshold & cast
                label_pred_th = np.where(
                    label_pred >= 0.5, 1.0, 0.0).astype(np.ubyte)

                # add to output list
                label_pred_th_list.append(label_pred_th)

                # conver to sitkImage to save
                pred_image = sitk.GetImageFromArray(label_pred_th)

                # copy image properties
                pred_image.SetSpacing(img_sampled.GetSpacing())
                pred_image.SetOrigin(img_sampled.GetOrigin())
                pred_image.SetDirection(img_sampled.GetDirection())

                # save image
                if out_dir_for_debug != None:
                    pred_image_path = f'{out_dir_for_debug}/Rectum.sampled.pred.{i}.mhd'
                    print('saving pred_image...', pred_image_path)
                    sitk.WriteImage(pred_image, pred_image_path,
                                    True)  # useCompression:True

                i += 1

        ##################################################
        # combine segments using grid_org_wrt_grid000_I

        # find the max shifts in I
        max_I = [-1, -1, -1]
        for (grid_coord, grid_org_wrt_grid000_I) in grid_list:
            max_I[0] = max(max_I[0], grid_org_wrt_grid000_I[0])
            max_I[1] = max(max_I[1], grid_org_wrt_grid000_I[1])
            max_I[2] = max(max_I[2], grid_org_wrt_grid000_I[2])

        max_I = np.array(max_I).astype(int)
        print('max_I=', max_I)

        combined_label_size = max_I + [grid_size] * 3
        print('combined_label_size=', combined_label_size)

        combined_label = np.zeros(
            [combined_label_size[2], combined_label_size[1], combined_label_size[0]])
        print('combine_label.shape=', combined_label.shape)

        n = 0
        for (grid_coord, grid_org_wrt_grid000_I) in grid_list:
            print('===========================')
            print('n=', n)
            label_pred_th = label_pred_th_list[n]

            print('grid_org_wrt_grid000_I=', grid_org_wrt_grid000_I)
            grid_org_wrt_grid000_I = np.array(
                grid_org_wrt_grid000_I).astype(int)
            i_org = grid_org_wrt_grid000_I[0]
            j_org = grid_org_wrt_grid000_I[1]
            k_org = grid_org_wrt_grid000_I[2]

            for k in range(label_pred.shape[0]):
                for j in range(label_pred.shape[1]):
                    for i in range(label_pred.shape[2]):
                        a = combined_label[k_org+k, j_org+j, i_org+i]
                        b = label_pred_th[k, j, i]
                        combined_label[k_org+k, j_org+j, i_org+i] = a+b
            n += 1

        # threshold & cast
        combined_label = np.where(
            combined_label >= 0.5, 1.0, 0.0).astype(np.ubyte)

        # conver to sitkImage to save
        pred_image = sitk.GetImageFromArray(combined_label)

        # copy image properties
        grid_coord_000, _ = grid_list[0]
        pred_image.SetSpacing(grid_coord_000.spacing)
        pred_image.SetOrigin(grid_coord_000.origin)
        pred_image.SetDirection(grid_coord_000.direction)

        # save image
        if out_file is not None:
            print('saving pred_image...', out_file)
            sitk.WriteImage(pred_image, out_file, True)  # useCompression:True

        return pred_image


    def test(self):

        test_dir = os.path.join(self.worker_dir, 'test')
        print('test_dir=', test_dir)
        if not os.path.exists(test_dir):
            print('creating dir=', test_dir)
            os.makedirs(test_dir)

        print('torch.__version__=', torch.__version__)

        input_image_size = self.param['Model']["input_image_size"]
        input_image_spacing = self.param['Model']["input_image_spacing"]

        ############
        # parameters
        samples_per_image = self.param['Train']['samples_per_image']
        #batch_size = train_p["batch_size"]

        #########
        # device
        device = self.device

        ############
        # model
        model = self.unet
        model = model.to(device)

        ###############################
        # download data if not exists
        try:
            test_files = self.download_dataset('TestSet')
        except Exception as e:
            print('Error - File download failed!')
            return 

        print(f'# of test images = {len(test_files)}')

        ###########
        # dataset
        test_ds = SegDataset3(input_list=test_files,
                               grid_size=input_image_size,
                               grid_spacing=input_image_spacing,
                               samples_per_image=samples_per_image,
                               transform=NormalizeCT())
        test_dlr = DataLoader(dataset=test_ds, batch_size=1, shuffle=False)

        # #################
        # # Mean Dice Loss
        # loss_file_path = os.path.join(test_dir,'dice_loss.csv')
        # write_line(loss_file_path, f'i, loss')

        # loss_func = DiceLoss()
        # N_test = len(test_dlr)
        # print('N_test=', N_test)
        # with torch.no_grad():
        #     sum_loss = 0.0
        #     for i, (images, labels) in enumerate(test_dlr):
        #         # batch (size = 1 for test)
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         outputs = model(images)
        #         loss = loss_func(outputs, labels)

        #         print(f'{i}/{len(test_dir)}, loss={loss.item()}')
        #         append_line(loss_file_path, f'{i}, {loss.item()}')

        #         sum_loss += loss
        #     mean_loss = sum_loss/N_test

        #     append_line(loss_file_path, f'mean, {mean_loss.item()}')
        #     print('mean_loss=', mean_loss.item())            

        ################
        # Segment Images
        for case in test_files:
            
            str_file = case['structure_mhd']
            img_file = case['image_mhd']
            
            # extract image id
            parts = split_directory_path(img_file)
            parent_folder_name = parts[len(parts)-2]
            img_id = parent_folder_name

            print('img_id=', img_id)

            out_dir = os.path.join(test_dir, img_id)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)    

            # segment
            seg_file = os.path.join(out_dir,'seg.mhd')
            print(f'segmenting image {img_file} to seg_file={seg_file}')
            self.segment(img_file=img_file, out_file=seg_file)
            
            # save seg info file
            case['seg_mhd']=seg_file
            p=Param(case)
            p.save_to_txt(os.path.join(out_dir, "seg.info"))
       


