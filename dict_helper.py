import json


def load_from_txt(param_file):
    dict = {}
    with open(param_file) as myfile:
        for line in myfile:
            line = line.strip()

            if line == '':
                continue

            if line[0] == '#':
                continue

            key, value = line.partition('=')[::2]
            dict[key.strip()] = value.strip()
    return dict


def save_to_txt(dict, param_file):
    file = open(param_file, "w")
    for key in dict:
        file.write(f'{key}={dict[key]}\n')
    file.close()

def load_from_json(json_file):
    dict = {}
    # read txt file
    file = open(json_file, 'r')
    txt = file.read()
    file.close()

    # string to object
    dict = json.loads(txt)

    # copy dictionary
    for key in dict:
        dict[key] = dict[key]

    return dict

def save_to_json(dict, json_file):
    json_str = json.dumps(dict)
    file = open(json_file, "w")
    file.write(json_str)
    file.close()


def has_key(dict, key):
    return key in dict.keys()

def get_item_as_float_array(dict, key, delim=','):
    string_array = dict[key].split(delim)
    return [float(x) for x in string_array]

def get_item_as_int_array(dict, key, delim=','):
    string_array = dict[key].split(delim)
    return [int(x) for x in string_array]

def cast_to_int(dict, key):
    if dict.has_key(key):
        dict[key] = int(dict[key])

def cast_to_bool(dict, key):
    if dict.has_key(key):
        dict[key] = bool(dict[key])

def cast_to_float(dict, key):
    if dict.has_key(key):
        dict[key] = float(dict[key])

def cast_to_int_array(dict, key, delim=','):
    if dict.has_key(key):
        dict[key] = dict.get_item_as_int_array(key, delim)

def cast_to_float_array(dict, key, delim=','):
    if dict.has_key(key):
        dict[key] = dict.get_item_as_float_array(key, delim)

def dict_to_string(dict, delim=',', delim_replaced_with='[comma]'):
    cols = []
    for key in dict.keys():
        value = dict[key]
        col = str(value)
        col = col.replace(delim, delim_replaced_with)
        cols.append(col)
    return delim.join(cols)

def dict_list_to_string_list(dict_list, delim=',', delim_replaced_with='[comma]'):
    
    if len(dict_list) == 0:
        return []

    lines = []

    # header line
    line = delim.join(dict_list[0].keys())    
    lines.append(line)

    for d in dict_list:
        lines.append(dict_to_string(d))
    
    return lines


