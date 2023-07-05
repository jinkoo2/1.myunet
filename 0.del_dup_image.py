# because of the course directory, sometimes, there are duplicate images under different courses.

from os import listdir, mkdir
from os.path import join, isfile, isdir, exists
import shutil
from helper import write_line, append_line, read_key_value_pairs, get_subdir_names

from datetime import datetime

# input param
dir_src = 'c:\\data'



pt_names = get_subdir_names(dir_src)
print('N=', len(pt_names))

for pt_name in pt_names:
    pt_dir = join(dir_src, pt_name)
    
    cs_names = get_subdir_names(pt_dir)
    if len(cs_names)<2:
        continue

    dict = {}
    for cs_name in cs_names:
        cs_dir = join(pt_dir, cs_name)
        img_names = get_subdir_names(cs_dir)
        for img_name in img_names:
            if hasattr(dict, img_name):
                dict[img_name].append(cs_name)
            else: 
                dict[img_name] = [cs_name]
                
    for img_name in dict.keys():
        
        # duplicate courses 
        cs_names = dict[img_name]

        # if only 1 course, skip
        if len(cs_names)<2:
            continue 

        # if >1, keep only 1, but remove the others
        print(pt_name+' has duplicate! Remove the duplicates!')

print('done')
