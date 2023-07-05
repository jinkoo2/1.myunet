# because of the course directory, sometimes, there are duplicate images under different courses.

from os import listdir, mkdir
from os.path import join, isfile, isdir, exists
import shutil
from helper import write_line, append_line, read_key_value_pairs, get_subdir_names

from datetime import datetime

# input param
dir_src = '.\\data1'
stat_file = join(dir_src, 'rectum.stat.csv')

case_names = get_subdir_names(dir_src)
print('N=', len(case_names))

write_line(stat_file, f'img_dir,size[0],size[1],size[2],volume')

for case_name in case_names:
    case_dir = join(dir_src, case_name)

    print(case_dir)

    rectum_info_path = join(case_dir, 'Rectum.info')

    dict = read_key_value_pairs(rectum_info_path)
    bbox_w = [float(s) for s in dict['bbox'].split(',')]

    minx, maxx, miny, maxy, minz, maxz = bbox_w

    size = [maxx-minx, maxy-miny, maxz-minz]

    try:
        volume = dict['volume']
        append_line(
            stat_file, f'{case_dir},{size[0]},{size[1]},{size[2]},{volume}')
    except:
        append_line(stat_file, f'{case_dir},{size[0]},{size[1]},{size[2]},-1')


print('done')
