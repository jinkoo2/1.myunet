import requests
import json
from export_data_parser import create_pt_json_files, post_pt_json_files_to_mongo

URL = "http://roweb2:8083/pts"


def get_pts():
    r = requests.get(url=URL)
    data = r.json()
    print(data)
    print('done')


def post_pts():
    data = {
        "Id": "000c1a61-226c-4757-b22d-db0aa6aabfll3b",
        "Sex": "Female",
        "DateOfBirth": "8/24/1956 12:00:00 AM",
        "cs_list": [{"Id": "1"}, {"Id": "2"}],
        "img_list": [],
        "ps_list": [
            {
                "Id": "T10_L1",
                "PlanType": "ExternalBeam",
                "StructureSet[dot]Id": "T_L SPINE_PEL",
            },
        ],
        "sset_list": []
    }

    headers = {'Content-Type': 'application/json',
               'Accept': 'application/json'}

    r = requests.post(url=URL, data=json.dumps(data), headers=headers)
    print(r.text)
    print('done')


def load_and_post():
    # read txt file
    json_file = 'C:/data/_sample_pt.json'
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
    print('done')


# post_pts()
# get_pts()
# load_and_post()
data_dir = 'C:/data'

# collect the info under the data_dir and create pt.json file for each patient folder
# create_pt_json_files(data_dir)

# collect the pt json files and post to mongo db.
post_pt_json_files_to_mongo(data_dir, URL)
