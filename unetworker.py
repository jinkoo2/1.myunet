import torch
from unet import UNet
from helper import add_a_dim_to_tensor
from dataset import SegDataset1, NormalizeCT
import numpy as np
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from helper import append_line, write_line
from image_helper import get_grid_list_to_cover_rect
from os import mkdir, remove, write
from os.path import join, exists
from param import param_from_json
import SimpleITK as sitk
from image_coord import image_coord
from dataset import sample_image
from rect import rect
from DiceLoss import DiceLoss


def str_to_None(str):
    if str.strip() == 'None' or str.strip() == 'none':
        return None
    else:
        return str


def get_loss_function(name):
    if name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'L1Loss':
        return nn.L1Loss()
    elif name == 'DiceLoss':
        return DiceLoss()
    else:
        raise "Invalid loss_func:"+name


def get_optimizer(name, model, learning_rate):
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise "Invalid optimizer:"+name


class UNetWorker():

    def __init__(self, worker_file):

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
        in_channels = p['in_channels']
        out_channels = p['out_channels']
        n_blocks = p['n_blocks']
        start_filters = p['start_filters']
        activation = str_to_None(p['activation'])
        normalization = str_to_None(p['normalization'])
        conv_mode = str_to_None(p['conv_mode'])
        input_image_size = p['input_image_size']
        dim = p['dim']
        up_mode = p['up_mode']
        final_activation = str_to_None(p['final_activation'])
        model_file = p['model_file']

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
        print('model_file=', model_file)

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

        if exists(model_file):
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

    def train(self):

        print('torch.__version__=', torch.__version__)

        input_image_size = self.param["input_image_size"]
        train_p = self.param["train"]

        ############
        # parameters
        learning_rate = train_p["learning_rate"]
        max_num_of_epochs_per_run = train_p["max_num_of_epochs_per_run"]
        batch_size = train_p["batch_size"]
        out_dir = train_p["out_dir"]
        data_train_dir = train_p["data_train_dir"]
        data_valid_dir = train_p["data_valid_dir"]
        optimizer = train_p["optimizer"]

        #########
        # device
        device = self.device

        # data_train_dir = '/gpfs/scratch/jinkokim/data/train'
        # data_valid_dir = '/gpfs/scratch/jinkokim/data/valid'

        # if not exists(data_train_dir):
        #     data_train_dir = './data/train'
        #     data_valid_dir = './data/valid'

        print('data_train_dir=', data_train_dir)
        print('data_valid_dir=', data_valid_dir)

        #data_train_dir = './data3/train'

        ###########
        # dataset
        train_ds = SegDataset1(dir=data_train_dir,
                               grid_size=input_image_size,
                               grid_spacing=input_image_size,
                               samples_per_image=1,
                               transform=NormalizeCT())
        train_dr = DataLoader(
            dataset=train_ds, batch_size=batch_size, shuffle=True)

        valid_ds = SegDataset1(dir=data_train_dir,
                               grid_size=input_image_size,
                               grid_spacing=input_image_size,
                               samples_per_image=1,
                               transform=NormalizeCT())
        valid_dr = DataLoader(dataset=valid_ds, batch_size=1, shuffle=False)

        ############
        # model
        model = self.unet
        model = model.to(device)

        epoch0 = train_p["current_epoch"]
        max_epoch = epoch0+max_num_of_epochs_per_run

        print('num_epochs=', max_num_of_epochs_per_run)

        ##############
        # print model
        W = input_image_size
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
        elif train_p["loss_func_train"] == 'L1Loss':
            loss_func_train = nn.L1Loss()
        elif train_p["loss_func_train"] == 'DiceLoss':
            loss_func_train = DiceLoss
        else:
            raise "Invalid loss_func_train:"+train_p["loss_func_train"]

        # validation loss function
        loss_func_train = get_loss_function(train_p["loss_func_train"])
        loss_func_valid = get_loss_function(train_p["loss_func_valid"])

        # optimizer
        optimizer = get_optimizer(train_p["optimizer"], model, learning_rate)

        # training itr log files
        itr_file_path = f'{out_dir}/itr.csv'
        itr_epoch_file_path = f'{out_dir}/itr.epoch.csv'

        # make out dir
        if not exists(out_dir):
            mkdir(out_dir)

        # if first training, add headers
        if epoch0 == 0:
            write_line(itr_file_path, f'epoch, i, loss')
            write_line(itr_epoch_file_path, f'epoch, validation loss')

        n_steps_per_epoch = len(train_dr)/batch_size

        for epoch in range(epoch0+1, max_epoch):
            for i, (images, labels) in enumerate(train_dr):
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
                    f'Epoch [{epoch+1}/{max_num_of_epochs_per_run}], Step [{i+1}/{n_steps_per_epoch}], Loss: {loss.item():.4f}')

            # calc mean loss over the validation dataset (each epoch save the validation loss)
            N_valid = len(valid_dr)
            validation_loss = 1000000000000000000.0  # a random large number
            with torch.no_grad():
                sum_loss_valid = 0.0
                for i, (images, labels) in enumerate(valid_dr):
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
                    f'Epoch [{epoch+1}/{max_num_of_epochs_per_run}], Mean Validation Loss L2: {mean_loss_valid.item():.4f}')

                # validation loss
                validation_loss = mean_loss_valid.item()

            # if the validation is smallest, save the model
            if validation_loss < train_p["min_validation_loss"]:

                # save the model
                model_file = f'{out_dir}/model.mdl'
                print('saving model'+model_file)
                torch.save(model.state_dict(), model_file)

                # save the current validation loss as minimum
                train_p["min_validation_loss"] = validation_loss
                train_p["selected_model_epoch"] = epoch
                self.param["model_file"] = model_file

            # save the current epoch
            train_p["current_epoch"] = epoch

            # update the param file
            self.param.save_to_json(self.worker_file)

            # stop, if there is a stop.txt file
            stop_file = join(out_dir, 'stop.txt')
            if exists(stop_file):
                print('stop.txt found! exiting training.')
                break

        print('Finished Training')

    def segment(self, img_file, roi_rect_w, out_file=None, out_dir_for_debug=None):
        print('========segment()=========')
        print('img_file=', img_file)
        print("roi_rect_w=", roi_rect_w)
        print("out_file=", out_file)

        ###################################################
        # get the list of sampling grid list for roi_rect_w
        print('get the list of sampling grid list for roi_rect_w')
        grid_size = self.param["input_image_size"]
        grid_spacing = self.param["input_image_spacing"]
        n_border_pixels = int(np.power(2, self.param["n_blocks"]))
        grid_list = get_grid_list_to_cover_rect(
            roi_rect_w, grid_size, grid_spacing, n_border_pixels)
        print('grid_list=', grid_list)

        ###############################
        # Segment using the grid list
        label_pred_th_list = []
        model = self.unet
        image_resample_background_pixel_value = self.param["image_resample_background_pixel_value"]
        i = 0
        with torch.no_grad():
            for (grid_coord, grid_org_wrt_grid000_I) in grid_list:
                print('======= pred for grid =============')
                print('i=', i)
                print('grid_coord=', grid_coord)
                print('grid_org_wrt_grid000_I=', grid_org_wrt_grid000_I)

                # segment for the grid
                ct_sampled_image_path = join(
                    out_dir_for_debug, f'CT.sampled.{i}.mhd') if out_dir_for_debug is not None else None
                ct_sampled = sample_image(
                    img_file, grid_coord, 3, image_resample_background_pixel_value, sitk.sitkLinear, ct_sampled_image_path)
                ct_np = sitk.GetArrayFromImage(ct_sampled).astype('float32')

                # scale image intensity
                factor = 1.0/150.0
                ct_np = 2.0/(1+np.exp(-ct_np*factor))-1.0

                # add color channel (1, because it's a gray, 1 color per pixel)
                size = list(ct_np.shape)
                # insert 1 to the 0th element (color channel)
                size.insert(0, 1)
                # insert 1 to the 0th element (batch channel)
                size.insert(0, 1)
                ct_np.resize(size)  # ex) [1,1,64,64,64]

                images = torch.from_numpy(ct_np)

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
                pred_image.SetSpacing(ct_sampled.GetSpacing())
                pred_image.SetOrigin(ct_sampled.GetOrigin())
                pred_image.SetDirection(ct_sampled.GetDirection())

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
