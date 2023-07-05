from param import Param, param_from_json
from db_service import load_db, get_contour_list, get_plan_list, uuid2pid
from dict_helper import dict_list_to_string_list, load_from_json
from os_helper import write_all_text
from datetime import datetime
from os.path import join
import os
import random

def make_cont_dataset(param_file):
    param = load_from_json(param_file)
    contour_list_json = param["contour_list_json"]
    name = param['name']
    filter_param_list = param["filter_list"]

    out_dir = param['out_dir']
    out_filename_header = param['out_filename_header']
    out_json = join(out_dir, out_filename_header+'.json')
    out_csv = join(out_dir, out_filename_header+'.csv')

    print('===== parameters =====')
    print('param_file', param_file)
    print('contour_list_json', contour_list_json)
    print('name', name)
    print('filter_param_list', filter_param_list)
    print('out_filename_header', out_filename_header)
    print('out_json', out_json)
    print('out_csv', out_csv)

    ##################
    # contour list file
    list = param_from_json(contour_list_json)["list"]

    print(f'N={len(list)} - all contours')

    #########
    # filter
    list = filter(list, filter_param_list)

    ###################
    # suffle the list
    random.shuffle(list)

    ###############################
    # div to train, valid, and test
    N = len(list)
    N_train = int(N * param['train_fraction'])
    N_valid = int(N * param['valid_fraction'])
    N_test = N - (N_train + N_valid)
    print("N=", N)
    print("N_train=", N_train)
    print("N_valid=", N_valid)
    print("N_test=", N_test)

    train_list = list[0:N_train-1]
    valid_list = list[N_train:(N_train+N_valid-1)]
    test_list = list[N_train+N_valid:]

    ######################
    # save seg dataset
    p = Param()
    p['name'] = name.lower()+'.global'
    p['description'] = f'recent {name} contours in Eclipse'
    p['count'] = len(list)
    p['created_by'] = 'jkim20'
    p['created_on'] = datetime.utcnow().isoformat()+'Z'
    p['train'] =  {'count': len(train_list),'list': train_list} 
    p['valid'] =  {'count': len(valid_list),'list': valid_list} 
    p['test'] =  {'count': len(test_list),'list': test_list} 
    p.save_to_json(out_json)

    lines = dict_list_to_string_list(list)
    print(f'saving to {out_csv}...')
    write_all_text(out_csv, '\n'.join(lines))


def filter(list, filter_param_list):
    for f in filter_param_list:
        if f['method'] == 'filter_strip_lower_eq':
            list = filter_strip_lower_eq(list, f['key'], f['value'])
        elif f['method'] == 'filter_eq':
            list = filter_eq(list, f['key'], f['value'])
        elif f['method'] == 'filter_mid_percent':
            list = filter_mid_percent(list, f['key'], f['value'])
        elif f['method'] == 'filter_strip_lower_contain':
            list = filter_strip_lower_contain(list, f['key'], f['value'])
        elif f['method'] == 'filter_strip_lower_in_value_list':
            list = filter_strip_lower_in_value_list(list, f['key'], f['value'])
        else:
            raise Exception('Unknown filter method:'+f['method'])

        print(f'N={len(list)} after filter by {f["key"]}')
    return list


def filter_strip_lower_eq(list, key, value):
    return [c for c in list if c[key].strip().lower() == value.strip().lower()]


def filter_strip_lower_contain(list, key, value):
    return [c for c in list if value.strip().lower() in c[key].strip().lower()]


def filter_strip_lower_in_value_list(list, key, value_list):
    return [c for c in list if c[key].strip().lower() in value_list]


def filter_eq(list, key, value):
    return [c for c in list if c[key] == value]


def filter_mid_percent(list, key, percent_to_keep):

    print(f'filter_mid_percent(list, {key}, {percent_to_keep}')

    # soft in place
    list.sort(key=lambda x: x[key], reverse=False)

    N = len(list)
    print(f'N={N}')

    half_to_remove = int(N * (0.5 * (1.0 - (percent_to_keep/100.0))))
    i_start = half_to_remove
    i_end = N - half_to_remove
    print(f'half_to_remove={half_to_remove}')
    print(f'i_start={i_start}')
    print(f'i_end={i_end}')

    return list[i_start:i_end]


param_dir = 'C:/data/_db/_make_cont_dataset_params'
print('param_dir=', param_dir)

for f in os.listdir(param_dir):
    param_file = join(param_dir, f)

    if not os.path.isfile(param_file):
        print(f'{param_file} is not a file. so skipping...')
        continue

    print()
    if os.path.splitext(param_file)[1].lower() == '.json':
        print('param_file=', param_file)
        make_cont_dataset(param_file)
    else:
        print(f'{param_file} ext is not ".json". so skipping...')
        continue
    
    
print('done')
