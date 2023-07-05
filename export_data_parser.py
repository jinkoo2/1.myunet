#
# utility functions to generate the db file based on the data in the directory (exported from ARIA by the exporter)
#

import os
from os.path import join
from param import param_from_txt, Param
import numpy as np
from helper import decode_path, s2f, s2i
import json
import requests

root_dir = 'c:\\data'


def dir_exists(dir):
    return os.path.exists(dir) and os.path.isdir(dir)


def file_exists(dir):
    return os.path.exists(dir) and os.path.isfile(dir)


pt_list = []


def load_pt_list(pt_list_dir):
    print(f'load_pt_list({pt_list_dir})')

    pt_list = []
    pt_dir_list_with_error = []

    if dir_exists(pt_list_dir):
        for pid in os.listdir(pt_list_dir):
            pt_dir = join(pt_list_dir, pid)
            if dir_exists(pt_dir):
                try:
                    pt = load_pt(pt_dir)
                    pt_list.append(pt)
                except:
                    pt_dir_list_with_error.append(pt_dir)
                    print(f'Error loading a patient{pt_dir}. Skipping...')

    return pt_list, pt_dir_list_with_error


def load_pt_json_and_post(pt_json_file, URL):
    print(f'load_pt_json_and_post({pt_json_file})')

    # read txt file
    json_file = pt_json_file
    file = open(json_file, 'r')
    txt = file.read()
    file.close()

    txt = txt.replace('Image.', 'Image[dot]')
    txt = txt.replace('MeshGeometry.', 'MeshGeometry[dot]')
    txt = txt.replace('Meterset.', 'Meterset[dot]')
    txt = txt.replace('ControlPoints.', 'ControlPoints[dot]')
    txt = txt.replace('Blocks.', 'Blocks[dot]')
    txt = txt.replace('Boluses.', 'Boluses[dot]')
    txt = txt.replace(
        'NaN', '-100000000000000000000000000000000000000000000000000000.0')
    txt = txt.replace('ExternalBeam.', 'ExternalBeam[dot]')
    txt = txt.replace('MLC.', 'MLC[dot]')
    txt = txt.replace('StructureSet.', 'StructureSet[dot]')
    txt = txt.replace('Series.', 'Series[dot]')
    txt = txt.replace('Course.', 'Course[dot]')

    # string to object
    dict = json.loads(txt)
    # dict['img_list']=[]
    # dict['ps_list']=[]
    # dict['sset_list']=[]
    #     {
    #         'Id': 'L_SPINE_PELV',
    #         'Comment': '',
    #         'HistoryDateTime': '12/12/2012 9:13:17 AM',
    #         'HistoryUserName': 'zhhan',
    #         'Image[dot]Id': 'CT_10102012',
    #         'UID': '1.2.246.352.71.4.920606082496.65756.20121212085935',
    #     },
    #     {
    #         'Id': 'T_L SPINE_PEL',
    #         'Comment': '',
    #         'HistoryDateTime': '12/12/2012 4:01:30 PM',
    #         'HistoryUserName': 'wcheng',
    #         'Image[dot]Id': 'CT_12112012',
    #         'UID': '1.2.246.352.71.4.920606082496.68361.20121212104711',
    #     }
    # ]

    # print(dict)
    headers = {'Content-Type': 'application/json',
               'Accept': 'application/json'}
    d = json.dumps(dict)
    print('dumps worked')
    r = requests.post(url=URL, data=d, headers=headers)
    print(r.text)
    # print('done')
    return r


def post_pt_json_files_to_mongo(pt_list_dir, URL):
    print(f'post_pt_json_files_to_mongo({pt_list_dir})')

    LEN_PID = len("3dcb0228-1487-4246-8dc1-81771327954f")
    pt_dir_list_with_error = []

    if dir_exists(pt_list_dir):
        dir_list = os.listdir(pt_list_dir)
        N = len(dir_list)
        for i, pid in enumerate(dir_list):
            print('===================')
            print(f'{i}/{N} {pid}')

            # check the length of pid
            if len(pid) != LEN_PID:
                print(
                    f'Invalid pid - the length much be {len("3dcb0228-1487-4246-8dc1-81771327954f")}. skipping.')
                continue

            pt_dir = join(pt_list_dir, pid)
            if dir_exists(pt_dir):
                try:
                    # skip if not dir
                    if os.path.isfile(pt_dir):
                        print(f'pt_dir={pt_dir} is not directory. skipping.')
                        continue

                    # skip of pt.json does not exist.
                    pt_json_file = join(pt_dir, 'pt.json')
                    if not os.path.exists(pt_json_file):
                        print('pt.json not found. skipping.')
                        continue

                    # load pt json and post
                    print('load_pt_json_and_post()')
                    res = load_pt_json_and_post(pt_json_file, URL)

                except Exception as exn:
                    print(exn)
                    pt_dir_list_with_error.append(f'{pt_dir} error:{exn}')
                    print(f'Error for {pt_dir}. Skipping...')

                except:
                    pt_dir_list_with_error.append(pt_dir)
                    print(f'Error for {pt_dir}. Skipping...')

    return pt_dir_list_with_error


def create_pt_json_files(pt_list_dir):
    print(f'create_pt_json_files({pt_list_dir})')

    LEN_PID = len("3dcb0228-1487-4246-8dc1-81771327954f")
    pt_json_file_list = []
    pt_dir_list_with_error = []

    if dir_exists(pt_list_dir):

        dir_list = os.listdir(pt_list_dir)
        N = len(dir_list)
        for i, pid in enumerate(dir_list):
            print('===================')
            print(f'{i}/{N} {pid}')

            # check the length of pid
            if len(pid) != LEN_PID:
                print(
                    f'Invalid pid - the length much be {len("3dcb0228-1487-4246-8dc1-81771327954f")}. skipping.')

            pt_dir = join(pt_list_dir, pid)
            if dir_exists(pt_dir):

                try:
                    # skip if not dir
                    if os.path.isfile(pt_dir):
                        print(f'pt_dir={pt_dir} is not directory. skipping.')
                        continue

                    # skip of pt.json exists.
                    pt_json_file = join(pt_dir, 'pt.json')
                    if os.path.exists(pt_json_file):
                        print('pt.json exists. skipping.')
                        continue

                    # load pt
                    pt = load_pt(pt_dir)

                    # save as json file

                    print(f"Saving...{pt_json_file}")
                    pt.save_to_json(pt_json_file)

                    pt_json_file_list.append(pt_json_file)

                except:
                    pt_dir_list_with_error.append(pt_dir)
                    print(f'Error loading a patient{pt_dir}. Skipping...')

    return pt_json_file_list, pt_dir_list_with_error


def load_pt(pt_dir):
    print(f'load_pt({pt_dir})')

    pt_info_file = join(pt_dir, 'info.txt')
    pt = param_from_txt(pt_info_file)

    # list course list
    pt["cs_list"] = load_cs_list(join(pt_dir, "cs_list"))
    pt["img_list"] = load_img_list(join(pt_dir, "img_list"))
    pt["ps_list"] = load_ps_list(join(pt_dir, "ps_list"))
    pt["sset_list"] = load_sset_list(join(pt_dir, "sset_list"))

    return pt


def load_cs_list(cs_list_dir):
    print(f'load_cs_list({cs_list_dir})')
    cs_list = []

    if dir_exists(cs_list_dir):
        for csid in os.listdir(cs_list_dir):
            cs_dir = join(cs_list_dir, csid)
            if dir_exists(cs_dir):
                cs = load_cs(cs_dir)
                cs_list.append(cs)

    return cs_list


def load_cs(cs_dir):

    print(f'load_cs({cs_dir})')

    cs_Id_encoded = os.path.split(cs_dir)[1]
    cs = {}
    cs["Id"] = decode_path(cs_Id_encoded)
    return cs


def load_img_list(img_list_dir):

    print(f'load_img_list({img_list_dir})')

    img_list = []

    if dir_exists(img_list_dir):
        for id in os.listdir(img_list_dir):
            img_dir = join(img_list_dir, id)
            if dir_exists(img_dir):
                img = load_img(img_dir)
                img_list.append(img)

    return img_list


def load_img(img_dir):

    print(f'load_img({img_dir})')

    info_file = join(img_dir, 'info.txt')
    img = param_from_txt(info_file)

    # convert types
    img["HasUserOrigin"] = bool(img["HasUserOrigin"])
    img["Level"] = int(img["Level"])
    img["Window"] = int(img["Window"])

    # image header
    img_mhd_file = join(img_dir, 'img.mhd')
    header = param_from_txt(img_mhd_file)
    header.cast_to_int("NDims")
    header.cast_to_bool("BinaryData")
    header.cast_to_bool("BinaryDataByteOrderMSB")
    header.cast_to_bool("CompressedData")
    header.cast_to_int("CompressedDataSize")
    header.cast_to_float_array("TransformMatrix", ' ')
    header.cast_to_float_array("Offset", ' ')
    header.cast_to_float_array("CenterOfRotation", ' ')
    header.cast_to_float_array("ElementSpacing", ' ')
    header.cast_to_int_array("DimSize", ' ')

    img["header"] = header

    # you can add some other derived information from the pixel data or the header (like size)

    # image physical length
    spacing = header['ElementSpacing']
    size = header['DimSize']
    length = np.array(spacing)*np.array(size)  # physical size
    img['physical_length'] = length.tolist()

    return img


def load_sset_list(sset_list_dir):

    print(f'load_sset_list({sset_list_dir})')

    sset_list = []

    if dir_exists(sset_list_dir):
        for id in os.listdir(sset_list_dir):
            sset_dir = join(sset_list_dir, id)
            if dir_exists(sset_dir):
                sset = load_sset(sset_dir)
                sset_list.append(sset)

    return sset_list


def load_sset(sset_dir):

    print(f'load_sset({sset_dir})')

    info_file = join(sset_dir, 'info.txt')
    sset = param_from_txt(info_file)

    # load structure list
    s_list = load_s_list(sset_dir)

    sset["s_list"] = s_list

    return sset


def load_s_list(s_list_dir):

    print(f'load_s_list({s_list_dir})')

    if dir_exists(s_list_dir) is not True:
        return []

    s_list = []

    # get info files (Body.info)
    s_info_filename_list = []
    if dir_exists(s_list_dir):
        for fname in os.listdir(s_list_dir):
            if fname.lower().endswith('.info'):
                s_info_filename_list.append(fname)

    for s_info_filename in s_info_filename_list:
        s_name_encoded = os.path.splitext(s_info_filename)[0]
        s = load_s(s_list_dir, s_name_encoded)
        if s is not None:
            s_list.append(s)

    return s_list


def load_s(s_list_dir, s_name_encoded):

    print(f'load_s({s_list_dir},{s_name_encoded})')

    info_file = join(s_list_dir, s_name_encoded)+".info"
    s = param_from_txt(info_file)

    # skip if a point (no volume), like BB
    if s["volume"].strip() == 'NaN':
        print('The structure volume is NaN, so skipping...')
        return None

    # name
    s["name"] = decode_path(s_name_encoded)

    # convert numbers
    s.cast_to_float_array("bbox", ',')
    s.cast_to_float("volume")
    s.cast_to_int("NumberOfSeparateParts")
    s.cast_to_int("ROINumber")
    s.cast_to_float_array("MeshGeometry.Bounds", ',')
    s.cast_to_bool("HasSegment")
    s.cast_to_bool("IsHighResolution")

    return s


def load_ps_list(ps_list_dir):

    print(f'load_ps_list({ps_list_dir})')

    ps_list = []

    if dir_exists(ps_list_dir):
        for id in os.listdir(ps_list_dir):
            ps_dir = join(ps_list_dir, id)
            if dir_exists(ps_dir):
                ps = load_ps(ps_dir)
                ps_list.append(ps)

    return ps_list


def load_ps(ps_dir):

    print(f'load_ps({ps_dir})')

    info_file = join(ps_dir, 'info.txt')
    ps = param_from_txt(info_file)

    # convert types
    ps.cast_to_bool("ApprovalStatus")
    ps.cast_to_int("NumberOfFractions")

    # load structure list
    beam_list = load_beam_list(ps_dir)

    ps["beam_list"] = beam_list

    return ps


def load_beam_list(beam_list_dir):

    print(f'load_beam_list({beam_list_dir})')

    if dir_exists(beam_list_dir) is not True:
        return []

    beam_list = []

    # get beam dirs (beam_1)
    beam_dirname_list = []
    for x in os.listdir(beam_list_dir):
        if x.lower().startswith('beam_'):
            beam_dirname_list.append(x)

    for beam_dir_name in beam_dirname_list:
        beam_dir = join(beam_list_dir, beam_dir_name)
        if dir_exists(beam_dir):
            beam = load_beam(beam_dir)
            beam_list.append(beam)

    return beam_list


def load_beam(beam_dir):

    print(f'load_beam({beam_dir})')

    info_file = join(beam_dir, 'info.txt')
    beam = param_from_txt(info_file)

    # convert types
    beam.cast_to_float('Meterset.Value')
    beam.cast_to_int('DoseRate')
    beam.cast_to_float('SSD')
    beam.cast_to_float('AverageSSD')
    try:
        beam.cast_to_float_array('IsocenterPosition', ',')
    except:
        print("IsocenterPosition cannot be converted to float array")
    beam.cast_to_float('WeightFactor')
    beam.cast_to_int('ControlPoints.Count')
    beam.cast_to_int('Blocks.Count')
    beam.cast_to_int('Boluses.Count')
    if beam['ControlPoints.Count'] > 0:
        N = beam['ControlPoints.Count']
        jaw_list = beam['ControlPoints.JawPositionsList'].split('|')
        c_list = s2f(beam['ControlPoints.CollimatorAngleList'].split('|'))
        g_list = s2f(beam['ControlPoints.GantryAngleList'].split('|'))
        mlc_list = beam['ControlPoints.LeafPositionsList'].split('|')
        mu_list = s2f(beam['ControlPoints.MetersetWeightList'].split('|'))
        rtn_list = s2f(
            beam['ControlPoints.PatientSupportAngleList'].split('|'))
        lat_list = s2f(
            beam['ControlPoints.TableTopLateralPositionList'].split('|'))
        lng_list = s2f(
            beam['ControlPoints.TableTopLongitudinalPositionList'].split('|'))
        vrt_list = s2f(
            beam['ControlPoints.TableTopVerticalPositionList'].split('|'))
        cp_list = []
        for n in range(N):
            cp = {}
            cp["JAW"] = s2f(jaw_list[n].replace('VRect(', '').replace(')', '').replace(
                'X1=', '').replace('Y1=', '').replace('X2=', '').replace('Y2=', '').split(','))
            cp["COL"] = c_list[n]
            cp["G"] = g_list[n]

            if mlc_list[n] is not '':
                banks = mlc_list[n].split('/')
                cp["MLC"] = {"Bank_A": s2f(banks[0].split(
                    ',')), "Bank_B": s2f(banks[1].split(','))}
            cp["t"] = mu_list[n]
            cp["RTN"] = rtn_list[n]
            cp["LAT"] = lat_list[n]
            cp["LNG"] = lng_list[n]
            cp["VRT"] = vrt_list[n]
            cp_list.append(cp)
    beam["ControlPoints"] = cp_list

    return beam


def parse_export_data(data_root_dir):

    print(f'parse_export_data({data_root_dir})')

    db = Param()
    db["root_dir"] = root_dir
    pt_list, pt_dir_list_with_error = load_pt_list(data_root_dir)
    db["pt_list"] = pt_list

    print('=========cases with error==============')
    print(pt_dir_list_with_error)
    print('=======================================')

    return db
