import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import SimpleITK as sitk
from os import listdir, mkdir, makedirs
from os.path import join, isfile, isdir, exists, split
import shutil

from rect import rect
from image_coord import image_coord
from helper import read_key_value_pairs, encode_path, download
from image_helper import sample_image, read_image
from param import Param, param_from_json

def is_case_dir(dir):
    # length must be 8
    if len(dir) != 8:
        return False

    # must be a number
    if not dir.isnumeric():
        return False

    return True


def print_image_prop(img, name):
    print('image:', name)
    print('================')
    print('origin = ', np.array(img.GetOrigin()))
    print('spacing =', np.array(img.GetSpacing()))
    print('size = ', np.array(img.GetSize()))


def read_and_normalize(img_path):
    # read image
    itkImage = read_image(img_path)

    org = itkImage.GetOrigin()
    spacing = itkImage.GetSpacing()
    size = itkImage.GetSize()
    direction = itkImage.GetDirection()

    # convert to numpy array (float32)
    npImage = sitk.GetArrayFromImage(itkImage).astype('float32')

    # normalize and convert to toTensor
    return torch.from_numpy(normalize(npImage)), image_coord(size=size, origin=org, spacing=spacing, direction=direction)

# return samples_per_image random samples within the organ rect.

# place rects randomly within the organ rect


class SegDataset1(Dataset):
    def __init__(self, dir, grid_size, grid_spacing, samples_per_image, transform=None, sampled_image_out_dir=None, input_image_resample_background_pixel_value=-1000):
        self.dir = dir
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing

        self.samples_per_image = samples_per_image
        self.transform = transform
        self.sampled_image_out_dir = sampled_image_out_dir

        self.input_image_resample_background_pixel_value = input_image_resample_background_pixel_value

        # read the directoy
        self.image_dirname_list = [
            item for item in filter(is_case_dir, listdir(dir))]

    def __getitem__(self, index):
        img_index = int(index/self.samples_per_image)
        case_name = self.image_dirname_list[img_index]
        case_dir = join(self.dir, case_name)
        ct_path = join(case_dir, 'CT.mhd')
        rectum_path = join(case_dir, 'Rectum.mhd')
        rectum_info_path = join(case_dir, 'Rectum.info')

        # print(ct_path)
        # print(rectum_path)
        # print(rectum_info_path)

        # read the bounding box of the rectum
        dict = read_key_value_pairs(rectum_info_path)
        bbox_w = [float(s) for s in dict['bbox'].split(',')]

        # print(bbox)

        minx, maxx, miny, maxy, minz, maxz = bbox_w

        organ_rect_w = rect(low=[minx, miny, minz], high=[maxx, maxy, maxz])

        # print('organ_bbox_o=', organ_bbox_o)

        #u_min = np.array([0.0, 0.0, 0.0])
        #u_max = np.array([1.0, 1.0, 1.0])
        # a random point in unit space
        #u = np.array([0.5, 0.5, 0.5])
        u = np.random.rand(3)

        # scale into bounding box, which will be the center of the sampling
        grid_center_phy = organ_rect_w.low + u * organ_rect_w.size()
        # print('grid_center=', grid_center_phy)

        grid_size_phy = [self.grid_size * self.grid_spacing] * 3
        # print('grid_size_phy=', grid_size_phy)

        # pick a random point in the bounding box
        grid_half_width = grid_size_phy * np.array([0.5, 0.5, 0.5])
        # print('grid_half_width=', grid_half_width)
        grid_org = grid_center_phy - grid_half_width
        grid_high = grid_center_phy + grid_half_width
        # print('grid_low=', grid_org)
        # print('grid_high=', grid_high)

        # sample the ct & rectum
        grid_coord = image_coord(origin=grid_org, size=[
                                 self.grid_size]*3, spacing=[self.grid_spacing]*3)

        # print('grid_coord=', grid_coord)

        ct_sampled_image_path = f'{self.sampled_image_out_dir}/CT_sample_{index}.mhd' if self.sampled_image_out_dir else None
        ct_sampled = sample_image(
            ct_path, grid_coord, 3, self.input_image_resample_background_pixel_value, sitk.sitkLinear, ct_sampled_image_path)

        rectum_sampled_image_path = f'{self.sampled_image_out_dir}/Rectum_sample_{index}.mhd' if self.sampled_image_out_dir else None
        rectum_sampled = sample_image(
            rectum_path, grid_coord, 3, 0, sitk.sitkNearestNeighbor, rectum_sampled_image_path)

        # sitImage to Numpy
        ct_np = sitk.GetArrayFromImage(ct_sampled).astype('float32')
        rectum_np = sitk.GetArrayFromImage(rectum_sampled).astype('float32')

        # add color channel (1, because it's a gray, 1 color per pixel)
        size = list(ct_np.shape)
        size.insert(0, 1)  # insert 1 to the 0th element
        ct_np.resize(size)
        rectum_np.resize(size)

        sample = ct_np, rectum_np

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.samples_per_image * len(self.image_dirname_list)

# sample CT and label image at the center of the organ rect.

# place a rect at the center of the organ rect


class SegDataset2(Dataset):
    def __init__(self, dir, grid_size, grid_spacing, transform=None, sampled_image_out_dir=None, input_image_resample_background_pixel_value=-1000):
        self.dir = dir
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing

        self.transform = transform
        self.sampled_image_out_dir = sampled_image_out_dir
        self.input_image_resample_background_pixel_value = input_image_resample_background_pixel_value

        # read the directoy
        self.image_dirname_list = [
            item for item in filter(is_case_dir, listdir(dir))]

    def __getitem__(self, index):
        img_index = index
        case_name = self.image_dirname_list[img_index]
        case_dir = join(self.dir, case_name)
        ct_path = join(case_dir, 'CT.mhd')
        rectum_path = join(case_dir, 'Rectum.mhd')
        rectum_info_path = join(case_dir, 'Rectum.info')

        # print(ct_path)
        # print(rectum_path)
        # print(rectum_info_path)

        # read the bounding box of the rectum
        dict = read_key_value_pairs(rectum_info_path)
        bbox_w = [float(s) for s in dict['bbox'].split(',')]

        # print(bbox)

        minx, maxx, miny, maxy, minz, maxz = bbox_w

        organ_rect_w = rect(low=[minx, miny, minz], high=[maxx, maxy, maxz])

        # print('organ_bbox_o=', organ_bbox_o)

        #u_min = np.array([0.0, 0.0, 0.0])
        #u_max = np.array([1.0, 1.0, 1.0])
        # center point in the u coordinate
        organ_center_u = np.array([0.5, 0.5, 0.5])

        # the resampleing grid center in w
        grid_center_w = organ_rect_w.low + organ_center_u * organ_rect_w.size()

        # print('grid_center=', grid_center_phy)

        grid_size_w = [self.grid_size * self.grid_spacing] * 3
        # print('grid_size_phy=', grid_size_phy)

        # pick a random point in the bounding box
        grid_half_size_w = grid_size_w * np.array([0.5, 0.5, 0.5])
        # print('grid_half_width=', grid_half_width)
        grid_org_w = grid_center_w - grid_half_size_w
        # grid_high = grid_center_w + grid_half_size_w
        # print('grid_low=', grid_org)
        # print('grid_high=', grid_high)

        # sample the ct & rectum
        grid_coord = image_coord(origin=grid_org_w, size=[
                                 self.grid_size]*3, spacing=[self.grid_spacing]*3)

        # print('grid_coord=', grid_coord)

        # sample CT
        ct_sampled_image_path = f'{self.sampled_image_out_dir}/CT_sample_{index}.mhd' if self.sampled_image_out_dir else None
        ct_sampled = sample_image(
            ct_path, grid_coord, 3, self.input_image_resample_background_pixel_value, sitk.sitkLinear, ct_sampled_image_path)

        # sample organ
        rectum_sampled_image_path = f'{self.sampled_image_out_dir}/Rectum_sample_{index}.mhd' if self.sampled_image_out_dir else None
        rectum_sampled = sample_image(
            rectum_path, grid_coord, 3, 0, sitk.sitkNearestNeighbor, rectum_sampled_image_path)

        # sitkImage to Numpy
        ct_np = sitk.GetArrayFromImage(ct_sampled).astype('float32')
        rectum_np = sitk.GetArrayFromImage(rectum_sampled).astype('float32')

        # add color channel (1, because it's a gray, 1 color per pixel)
        size = list(ct_np.shape)
        size.insert(0, 1)  # insert 1 to the 0th element
        ct_np.resize(size)
        rectum_np.resize(size)

        sample = ct_np, rectum_np

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_dirname_list)

# normalize CT value


def normalize(input):
    factor = 1.0/150.0
    return 2.0/(1+np.exp(-input*factor))-1.0


class NormalizeCT:
    def __call__(self, sample):
        if type(sample) is tuple:
            inputs, targets = sample

            # the coefficient of 1/150 will map the range of -300 to 300 to good portion of the outputs;
            # outputs will be in [-1.0, 1.0]
            # factor = 1.0/150.0
            # 2.0/(1+np.exp(-inputs*factor))-1.0
            inputs = normalize(inputs)

            # convert to tensors and return
            return torch.from_numpy(inputs), torch.from_numpy(targets)
        else:
            # assume the sample numpy array image
            return normalize(sample)


class LocDataset1(Dataset):
    def __init__(self, dir, downsample_grid_size=64, transform=None, sampled_image_out_dir=None, input_image_resample_background_pixel_value=-1000):
        self.dir = dir
        self.downsample_grid_size = downsample_grid_size
        self.transform = transform
        self.sampled_image_out_dir = sampled_image_out_dir
        self.input_image_resample_background_pixel_value = input_image_resample_background_pixel_value

        # read the directoy
        self.image_dirname_list = [
            item for item in filter(is_case_dir, listdir(dir))]

        if self.sampled_image_out_dir:
            if not exists(self.sampled_image_out_dir):
                mkdir(self.sampled_image_out_dir)

    def get_item(self, index):
        # check if the index is valid?
        case_name = self.image_dirname_list[index]
        case_dir = join(self.dir, case_name)
        ct_path = join(case_dir, 'CT.mhd')
        rectum_info_path = join(case_dir, 'Rectum.info')

        print(ct_path)
        # print(rectum_path)
        # print(rectum_info_path)

        # read the bounding box of the rectum
        dict = read_key_value_pairs(rectum_info_path)
        bbox_w = [float(s) for s in dict['bbox'].split(',')]
        minx, maxx, miny, maxy, minz, maxz = bbox_w
        organ_rect_w = rect(low=[minx, miny, minz], high=[maxx, maxy, maxz])

        # print('bbox=', bbox)
        # print('organ_bbox_o=', organ_bbox_o)
        # print('organ_bbox_o.size()=', organ_bbox_o.size())

        # read the image
        reader = sitk.ImageFileReader()
        reader.SetImageIO("MetaImageIO")
        reader.SetFileName(ct_path)
        img = reader.Execute()

        # image bounding box
        img_coord = image_coord(origin=img.GetOrigin(), size=img.GetSize(
        ), spacing=img.GetSpacing(), direction=img.GetDirection())
        # print('img_coord=', img_coord)
        # print('img_coord.rect_o=', img_coord.rect_o())
        # print('img_coord.rect_o.size()=', img_coord.rect_o().size())

        # downsample the image to downsample_grid_size (ex, 64, 64, 64),
        sample_grid_coord = image_coord(
            origin=img.GetOrigin(),
            size=[self.downsample_grid_size] * 3,
            spacing=img_coord.rect_o().size(
            )/np.array([self.downsample_grid_size] * 3),
            direction=img_coord.direction
        )

        #print('sample_grid_coord', sample_grid_coord)

        # resample image
        ct_sampled_image_path = f'{self.sampled_image_out_dir}/CT_sample_{index}.mhd' if self.sampled_image_out_dir else None
        ct_sampled = sample_image(
            ct_path, sample_grid_coord, 0, self.input_image_resample_background_pixel_value, sitk.sitkLinear, ct_sampled_image_path)

        # normalize the organ bounding box
        organ_bbox_u = rect(low=img_coord.w2u(organ_rect_w.low),
                            high=img_coord.w2u(organ_rect_w.high))
        print(organ_bbox_u)
        center_u = (organ_bbox_u.low+organ_bbox_u.high)/np.array([2.0] * 3)
        size_u = organ_bbox_u.size()

        # print('organ_bbox_u=', organ_bbox_u)
        # print('center_u=', center_u)
        # print('size_u=', size_u)

        # sitImage to Numpy
        ct_np = sitk.GetArrayFromImage(ct_sampled).astype('float32')

        # add color channel (1, because it's a gray, 1 color per pixel)
        size = list(ct_np.shape)
        size.insert(0, 1)  # insert 1 to the 0th element
        ct_np.resize(size)

        # sample
        sample = ct_np, np.concatenate(
            (center_u, size_u), axis=0).astype('float32')

        if self.transform:
            sample = self.transform(sample)

        # bbox to obect coordinate system
        organ_bbox_o = rect(low=img_coord.w2o(organ_rect_w.low),
                            high=img_coord.w2o(organ_rect_w.high))
        print(organ_bbox_u)

        return sample, img_coord, organ_bbox_o

    def __getitem__(self, index):
        sample, _, _ = self.get_item(index)
        return sample

    def __len__(self):
        return len(self.image_dirname_list)


server_url = 'http://localhost:3000'
local_data_root = './local_data'
def download_file_if_not_exits(url, dst_file):
    
    if exists(dst_file):
        return

    #print(f'download_file_if_not_exits({url},{dst_file})')
    
    # check if the url exists
    
    # create dst dir if not exists
    dst_dir = split(dst_file)[0]
    if not exists(dst_dir):
        #print(f'makedirs({dst_dir})')
        makedirs(dst_dir)

    # download    
    download(url, dst_file)

def download_image_if_not_exist(dir_part):
    for ext in ["mhd", "zraw"]:
        url = f'{server_url}/{dir_part}/img.{ext}'
        dst_file = f'{local_data_root}/{dir_part}/img.{ext}'
        download_file_if_not_exits(url, dst_file)

def download_cont_if_not_exist(dir_part, name):
    for ext in ["mhd", "zraw", "info"]:
        url = f'{server_url}/{dir_part}/{encode_path(name)}.{ext}'
        dst_file = f'{local_data_root}/{dir_part}/{encode_path(name)}.{ext}'
        download_file_if_not_exits(url, dst_file)

def download_cont_info_file_if_not_exist(dir_part, name):
    url = f'{server_url}/{dir_part}/{encode_path(name)}.info'
    dst_file = f'{local_data_root}/{dir_part}/{encode_path(name)}.info'
    download_file_if_not_exits(url, dst_file)

class LocDataset2(Dataset):
    def __init__(self, dataset_json_file, data_for, downsample_grid_size=64, transform=None, sampled_image_out_dir=None, input_image_resample_background_pixel_value=-1000):
        self.dataset_json_file = dataset_json_file
        self.downsample_grid_size = downsample_grid_size
        self.transform = transform
        self.sampled_image_out_dir = sampled_image_out_dir
        self.input_image_resample_background_pixel_value = input_image_resample_background_pixel_value

        # if self.sampled_image_out_dir:
        #     if not exists(self.sampled_image_out_dir):
        #         mkdir(self.sampled_image_out_dir)

        # read the directoy
        # self.image_dirname_list = [
        #     item for item in filter(is_case_dir, listdir(dir))]
        # download data if not present in the local data dir
        data_dir = './local_data'

        #dataset
        print('dataset_json_file=', dataset_json_file)
        p_ds = param_from_json(dataset_json_file)

        print('[dataset]')
        print('dataset.name=', p_ds['name'])
        print('dataset.description=', p_ds['description'])
        print('dataset.count=', p_ds['count'])
        print('dataset.created_by=', p_ds['created_by'])
        print('dataset.created_on=', p_ds['created_on'])
        print('dataset.train.count=', p_ds['train']['count'])
        print('dataset.valid.count=', p_ds['valid']['count'])
        print('dataset.test.count=', p_ds['test']['count'])

        print('data_for=', data_for)

        cont_list = []
        if data_for=='train':
            cont_list = p_ds['train']['list']
        elif data_for=='valid':
            cont_list = p_ds['valid']['list']
        elif data_for=='test':
            cont_list = p_ds['test']['list']
        else:
            raise Exception('Invalid param - "data_for" - given {data_for}')

        if len(cont_list)==0:
            raise Exception('There is no contour in the contour list!')

        print('cont_list.count=', len(cont_list))

        samples = []
        
        N = len(cont_list)
        i=0
        for cont in cont_list:
            name = cont['Name']
            pid = cont['Patient.Id']
            imgid = cont['Image.Id']
            #psid = cont['PlanSetup.Id']
            ssetid = cont['StructureSet.Id']

            cont_dir_part = f'{pid}/sset_list/{encode_path(ssetid)}'
            img_dir_part = f'{pid}/img_list/{encode_path(imgid)}'

            print(f'[{i}/{N}] {name}')

            download_cont_info_file_if_not_exist(cont_dir_part, name)
            download_image_if_not_exist(img_dir_part)

            sample = {
                'name': name,
                'cont_dir': f'{local_data_root}/{cont_dir_part}',
                'img_dir':f'{local_data_root}/{img_dir_part}',
                'cont_info': cont
            }
            samples.append(sample)
            i+=1
            
        self.case_list = samples

    def get_item(self, index):
        # check if the index is valid?
        case = self.case_list[index]
        
        # image and cont info path
        img_path = join(case['img_dir'],'img.mhd')
        cont_info_path = join(case['cont_dir'], encode_path(case['name'])+'.info')
        if not exists(img_path):
            raise Exception(f'image not found - {img_path}')
        if not exists(cont_info_path):
            raise Exception(f'contour info file not found - {cont_info_path}')

        print(img_path)
        print(cont_info_path)
        
        # read the bounding box of the rectum
        dict = read_key_value_pairs(cont_info_path)
        bbox_w = [float(s) for s in dict['bbox'].split(',')]
        minx, maxx, miny, maxy, minz, maxz = bbox_w
        organ_rect_w = rect(low=[minx, miny, minz], high=[maxx, maxy, maxz])

        print('bbox_w=', bbox_w)
        print('organ_rect_w=', organ_rect_w)
        print('organ_rect_w.size()=', organ_rect_w.size())

        # read the image
        reader = sitk.ImageFileReader()
        reader.SetImageIO("MetaImageIO")
        reader.SetFileName(img_path)
        img = reader.Execute()

        # image bounding box
        img_coord = image_coord(origin=img.GetOrigin(), size=img.GetSize(
        ), spacing=img.GetSpacing(), direction=img.GetDirection())
        # print('img_coord=', img_coord)
        # print('img_coord.rect_o=', img_coord.rect_o())
        # print('img_coord.rect_o.size()=', img_coord.rect_o().size())

        # downsample the image to downsample_grid_size (ex, 64, 64, 64),
        sample_grid_coord = image_coord(
            origin=img.GetOrigin(),
            size=[self.downsample_grid_size] * 3,
            spacing=img_coord.rect_o().size(
            )/np.array([self.downsample_grid_size] * 3),
            direction=img_coord.direction
        )

        print('sample_grid_coord', sample_grid_coord)

        # resample image
        ct_sampled_image_path = f'{self.sampled_image_out_dir}/CT_sample_{index}.mhd' if self.sampled_image_out_dir else None
        ct_sampled = sample_image(
            img_path, sample_grid_coord, 0, self.input_image_resample_background_pixel_value, sitk.sitkLinear, ct_sampled_image_path)

        # normalize the organ bounding box
        organ_bbox_u = rect(low=img_coord.w2u(organ_rect_w.low),
                            high=img_coord.w2u(organ_rect_w.high))
        print(organ_bbox_u)
        center_u = (organ_bbox_u.low+organ_bbox_u.high)/np.array([2.0] * 3)
        size_u = organ_bbox_u.size()

        print('organ_bbox_u=', organ_bbox_u)
        print('center_u=', center_u)
        print('size_u=', size_u)

        # sitImage to Numpy
        ct_np = sitk.GetArrayFromImage(ct_sampled).astype('float32')

        # add color channel (1, because it's a gray, 1 color per pixel)
        size = list(ct_np.shape)
        size.insert(0, 1)  # insert 1 to the 0th element
        ct_np.resize(size)

        # sample
        sample = ct_np, np.concatenate(
            (center_u, size_u), axis=0).astype('float32')

        if self.transform:
            sample = self.transform(sample)

        # bbox to obect coordinate system
        organ_bbox_o = rect(low=img_coord.w2o(organ_rect_w.low),
                            high=img_coord.w2o(organ_rect_w.high))
        print(organ_bbox_o)

        return sample, img_coord, organ_bbox_o

    def __getitem__(self, index):
        sample, _, _ = self.get_item(index)
        return sample

    def __len__(self):
        return len(self.case_list)
