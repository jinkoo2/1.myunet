# because of the course directory, sometimes, there are duplicate images under different courses.

from os import listdir, mkdir
from os.path import join, isfile, isdir, exists
import shutil
from helper import write_line, append_line, read_key_value_pairs

from datetime import datetime

# input param
dir_src = 'c:\\data'
stat_file = join(dir_src, 'rectum.stat.csv')

def get_subdir_names(dir):
    return [x for x in listdir(dir) if isdir(join(dir,x))]

pt_names = get_subdir_names(dir_src)
print('N=', len(pt_names))

write_line(stat_file, f'img_dir,size[0],size[1],size[2],volume')

for pt_name in pt_names:
    pt_dir = join(dir_src, pt_name)
    
    for cs_name in get_subdir_names(pt_dir):
        cs_dir = join(pt_dir, cs_name)
        
        for img_name in get_subdir_names(cs_dir):
            img_dir = join(cs_dir, img_name)

            print(img_dir)

            rectum_info_path = join(img_dir, 'Rectum.info')
            
            dict = read_key_value_pairs(rectum_info_path)
            bbox_w = [float(s) for s in dict['bbox'].split(',')]

            minx, maxx, miny, maxy, minz, maxz = bbox_w

            size = [maxx-minx, maxy-miny, maxz-minz ]
            
            try:
                volume = dict['volume']
                append_line(stat_file, f'{img_dir},{size[0]},{size[1]},{size[2]},{volume}')
            except:
                append_line(stat_file, f'{img_dir},{size[0]},{size[1]},{size[2]},-1')


print('done')
