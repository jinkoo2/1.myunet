#generate orthogonal images of CTs
import SimpleITK as sitk
from os import listdir, mkdir
from os.path import join, isfile, isdir, exists

data_dir = '/gpfs/scratch/jinkokim/data/train'
if not exists(data_dir):
    data_dir = './data/train'

print('data_dir=', data_dir)

def extract_save(img, ext_size, ext_index, out_file):
    slice = sitk.Extract(img, size=ext_size, index=ext_index)
    slice = sitk.IntensityWindowing(slice, windowMinimum=-300.0, windowMaximum=300.0, outputMinimum=0.0, outputMaximum=255.0)
    slice = sitk.Cast(slice, sitk.sitkUInt8)
    print('out_file=', out_file)
    sitk.WriteImage(slice, out_file, True)  # useCompression:True

def make_ortho_images(img_dir):
    img_path = join(img_dir, "CT.mhd")
    print('img_path=', img_path)
    
    # check if image exists
    if not exists(img_path):
        print('Img not found. Skipping...,', img_path)
        return

    # read image information
    # image file reader
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(img_path)

    # read the image information without reading the bulk data
    file_reader.ReadImageInformation()
    size=file_reader.GetSize()
    img = file_reader.Execute()

    # extract x-y slice
    ext_size = [size[0], size[1], 0]
    ext_index = [0,0, int(size[2]/2)]
    out_file = img_path+'.xy.png'
    extract_save(img, ext_size, ext_index, out_file)

    # extract y-z slice
    ext_size = [0, size[1], size[2]]
    ext_index = [int(size[0]/2), 0, 0]
    out_file = img_path+'.yz.png'
    extract_save(img, ext_size, ext_index, out_file)

    # extract z-x slice
    ext_size = [size[0], 0, size[2]]
    ext_index = [0, int(size[1]/2), 0]
    out_file = img_path+'.zx.png'
    extract_save(img, ext_size, ext_index, out_file)
    
   
for dirname in listdir(data_dir):
    img_dir = join(data_dir, dirname)
    print('img_dir=', img_dir)
    make_ortho_images(img_dir)

print('done')