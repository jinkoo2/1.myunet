
'''
copy a small set of data from Rectum dataset
'''
import SimpleITK as sitk
from os import listdir, mkdir
from os.path import join, isfile, isdir, exists
import shutil
from helper import write_line, append_line, read_key_value_pairs, get_subdir_names

from datetime import datetime


# input param
num_samples = 5
dir_src = 'c:\\data'
dir_dst = 'C:\\Users\\jkim20\\Desktop\\projects\\pytorch\\1.myunet\\data1'


def copyimage(dir_in, img_name_in, dir_out, img_name_out):
    print(f'copyimage({dir_in}, {img_name_in}, {dir_out}, {img_name_out})')
    img_path_in = join(dir_in, img_name_in)
    img_path_out = join(dir_out, img_name_out)

    # source file not found, exit.
    if not isfile(img_path_in):
        print('File not found, skiping...: ', img_path_in)
        return False

    # desination file found, exit
    if isfile(img_name_out):
        print('File already exists, skiping...: ', img_path_out)
        return False

    # read
    img = sitk.ReadImage(img_path_in)

    # create dst directory
    if not isdir(dir_out):
        print('mkdir=>', dir_out)
        mkdir(dir_out)

    print('write=>', img_path_out)
    sitk.WriteImage(img, img_path_out, True)  # useCompression:True

    return True


def copyfile(dir_in, name_in, dir_out, name_out):
    print(f'copyfile({dir_in}, {name_in}, {dir_out}, {name_out})')
    path_in = join(dir_in, name_in)
    path_out = join(dir_out, name_out)

    # source file not found, exit.
    if not isfile(path_in):
        print('File not found, skiping...: ', path_in)
        return False

    # desination file found, exit
    if isfile(name_out):
        print('File already exists, skiping...: ', path_out)
        return False

    # create dst directory
    if not isdir(dir_out):
        print('mkdir=>', dir_out)
        mkdir(dir_out)

    # copy file
    print(f'copyfile({path_in}, {path_out})')
    shutil.copyfile(path_in, path_out)

    return True


def all_exists(dir, filename_list):
    for filename in filename_list:
        if not exists(join(dir, filename)):
            return False
    return True


files_to_copy = ['CT.mhd', 'CT.raw', 'Rectum.mhd',
                 'Rectum.raw', 'Rectum.info', 'exporter.info.txt']

# get start index
max_index = -1
filenames = get_subdir_names(dir_dst)
for filename in filenames:
    try:
        i = int(filename)
        if i > max_index:
            max_index = i
    except:
        print('invalid integer: ', filename)

print('max_index=', max_index)

# start index from the max index
i = max_index+1
print('starting index = ', i)

# for pt_name in listdir(dir_src)[:num_samples]:
for pt_name in get_subdir_names(dir_src):
    pt_dir = join(dir_src, pt_name)
    # print(f'pt={pt_name}')
    for cs_name in listdir(pt_dir):
        cs_dir = join(pt_dir, cs_name)
        # print(f'\tcs={cs_name}')
        for img_name in listdir(cs_dir):

            print(f'=== [ i={i} ] ===')
            img_dir = join(cs_dir, img_name)
            print(f'img_dir={img_dir}')

            # check if this has been copied already
            src_status_file = join(img_dir, 'copied.txt')
            if exists(src_status_file):
                print('copied earlier.. so skipping')
                continue

            # check all files available before start copying
            if not all_exists(img_dir, files_to_copy):
                print('not all files exist to copy. so skipping...')
                continue

            # if ImagingOrientation is not HeadFirstSuppine, skip
            # the current exporter cannot handle other orientation properly.
            # ImagingOrientation=HeadFirstSupine
            exporter_info_file = join(img_dir, 'exporter.info.txt')
            dict = read_key_value_pairs(exporter_info_file)
            try:
                if (dict['ImagingOrientation'].strip() != 'HeadFirstSupine'):
                    print('ImagingOrientation is not HeadFirstSupine, so skipping...')
                    continue
                if (dict['TreatmentOrientation'].strip() != 'HeadFirstSupine'):
                    print('TreatmentOrientation is not HeadFirstSupine, so skipping...')
                    continue
                if (dict['PlanType'].strip() != 'ExternalBeam'):
                    print('PlanType is not ExternalBeam, so skipping...')
                    continue
            except:
                continue            

            # check rectum info
            rectum_info_file = join(img_dir, 'Rectum.info')
            dict = read_key_value_pairs(rectum_info_file)
            try:
                volume = float(dict['volume'])
                if volume<35.0:
                    print('volume is too small, skipping')    
                    continue
                if volume>203.0:
                    print('volume is too large, skipping')    
                    continue

                NumberOfSeparateParts = int(dict['NumberOfSeparateParts'])
                if NumberOfSeparateParts is not 1:
                    print('NumberOfSeparateParts should be 1, skipping...')    
                    continue
            except:
                print('volume or NumberOfSeparateParts not found in Rectum.info file')
                continue 
            
            # check image info (length along z)
            dict = read_key_value_pairs(join(img_dir, 'CT.mhd'))
            try:
                spacing_z = float(dict['ElementSpacing'].strip()[2])
                size_z = float(dict['DimSize'].strip()[2])
                length_z = spacing_z * size_z
                if length_z> 500:
                    print('Image is too long for this study')
                    continue
            except:
                print('Image info not found in CT.mhd file')
                continue 

            # desination dir
            ct_dir_dst = join(dir_dst, str(i).zfill(8))
            dst_info_file = join(ct_dir_dst, 'info.txt')

            # copy CT.mhd (compress)
            if not copyimage(img_dir, 'CT.mhd', ct_dir_dst, 'CT.mhd'):
                continue

            # copy Rectum.mhd (compress)
            copyimage(img_dir, 'Rectum.mhd', ct_dir_dst, 'Rectum.mhd')

            # copy Rectum.info
            copyfile(img_dir, 'Rectum.info', ct_dir_dst, 'Rectum.info')

             # copy exporter.info.txt
            copyfile(img_dir, 'exporter.info.txt', ct_dir_dst, 'exporter.info.txt')

            # flag this folder is done
            write_line(src_status_file, f'copied_at={datetime.now()}')

            # save the source in the destination folder
            append_line(dst_info_file, f'src_img_dir={img_dir}')

            # increment index
            i += 1

print('done')
