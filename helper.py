from unet import UNet
from os import listdir, mkdir
from os.path import join, isfile, isdir, exists
import torch
from os import listdir
from os.path import join
from unet import UNet
from locnet import LocNet
import datetime
from image_coord import image_coord
import urllib.request

def to_str_array(arr):
    return [str(s) for s in arr]


def join_string_array(str_array):
    separator = ','
    return separator.join(str_array)


def array_to_string(arr):
    return join_string_array(to_str_array(arr))


def append_line(file, line):
    # Append-adds at last
    file1 = open(file, "a")  # append mode
    file1.write(line+'\n')
    file1.close()


def write_line(file, line):
    # Append-adds at last
    file1 = open(file, "w")  # write mode
    file1.write(line+'\n')
    file1.close()


def read_all_text(file):
    file = open("./_secure/private_key.pem", mode='r')
    txt = file.read()
    file.close()
    return txt


def add_a_dim_to_tensor(x):
    size = list(x.shape)
    size.insert(0, 1)  # insert 1 to the 0th element
    x.resize_(size)


def get_subdir_names(dir):
    return [x for x in listdir(dir) if isdir(join(dir, x))]


def read_key_value_pairs(file):
    dict = {}
    with open(file) as myfile:
        for line in myfile:
            key, value = line.partition('=')[::2]
            dict[key.strip()] = value.strip()
    return dict


def get_latest_model_unet1(out_dir):

    model = UNet(in_channels=1,
                 out_channels=1,
                 n_blocks=3,
                 start_filters=32,
                 activation='relu',
                 normalization='batch',
                 conv_mode='same',
                 dim=3)

    if not exists(out_dir):
        return model, 0

    model_files = [fname for fname in listdir(
        out_dir) if fname.startswith('model_') and fname.endswith('.mdl')]
    if len(model_files) > 0:
        model_epochs = [int(fname.split('.')[0].split('_')[1])
                        for fname in model_files]
        print('model_epochs=', model_epochs)
        max_epoch = max(model_epochs)
        latest_model_fname = join(out_dir, f'model_{max_epoch}.mdl')
        print('loading model from... ', latest_model_fname)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(latest_model_fname))
        else:
            model.load_state_dict(torch.load(
                latest_model_fname, map_location=torch.device('cpu')))

        return model, max_epoch
    else:
        print('no model file found. ')
        return model, 0


def get_latest_model_unet2(out_dir):

    model = UNet(in_channels=1,
                 out_channels=1,
                 n_blocks=4,
                 start_filters=8,
                 activation='relu',
                 normalization='batch',
                 conv_mode='same',
                 dim=3)

    if not exists(out_dir):
        return model, 0

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

        return model, max_epoch
    else:
        print('no model file found. ')
        return model, 0


def get_latest_model_locnet1(out_dir, input_image_size=64):

    model = LocNet(in_channels=1,
                   out_channels=1,
                   n_blocks=2,
                   start_filters=8,
                   activation='relu',
                   normalization=None,
                   conv_mode='same',
                   input_image_size=input_image_size,
                   dim=3)

    if not exists(out_dir):
        return model, 0

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

        return model, max_epoch
    else:
        print('no model file found. ')
        return model, 0


def get_latest_model_locnet2(out_dir, input_image_size=64):

    model = LocNet(in_channels=1,
                   out_channels=1,
                   n_blocks=2,
                   start_filters=16,
                   activation='relu',
                   normalization='batch',
                   conv_mode='same',
                   input_image_size=input_image_size,
                   dim=3)

    if not exists(out_dir):
        return model, 0

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

        return model, max_epoch
    else:
        print('no model file found. ')
        return model, 0


def get_latest_model_locnet3(out_dir, input_image_size=64):

    model = LocNet(in_channels=1,
                   out_channels=1,
                   n_blocks=2,
                   start_filters=32,
                   activation='relu',
                   normalization='batch',
                   conv_mode='same',
                   input_image_size=input_image_size,
                   dim=3)

    if not exists(out_dir):
        return model, 0

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

        return model, max_epoch
    else:
        print('no model file found. ')
        return model, 0


def s2f(str_arr):
    return [float(x) for x in str_arr]


def s2i(str_arr):
    return [int(x) for x in str_arr]


def now_str_utc():
    return f'{datetime.datetime.now(datetime.timezone.utc)}'


def max_of_two(a, b):
    if a > b:
        return a
    else:
        return b


# encode/decode should match with the exporter (which encodes stuff)
def encode_path(path0):
    return path0.replace(" ", "[sp]").replace("#", "[srp]").replace(".", "[dot]").replace("/", "[slsh]").replace("\\", "[bslsh]").replace("-", "[mns]")

def decode_path(path0):
    return path0.replace("[sp]", " ").replace("[srp]", "#").replace("[dot]", ".").replace("[slsh]", "/").replace("[bslsh]", "\\").replace("[mns]", "-")


def download(url, dest_file):
    print(f'{url}-->{dest_file}')
    urllib.request.urlretrieve(url, dest_file)



