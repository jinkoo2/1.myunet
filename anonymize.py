# remove pt info from the folder
import os
import uuid

from numpy.lib.function_base import append
from param import param_from_txt, param_from_json, Param
from helper import append_line



def is_mrn(name):
    # has to be 8 digit
    if len(name) != 8:
        return False
    
    # has to be int number
    try:
        # check if integer number
        pid_int = int(name)
        return True
    except:
        return False

#
# parse directory structure, remove the pt id.
# save the pt id and uuid into .json file
#

def anonymize():
    root_dir = "c:\\data"
    err_file = "c:\\data\\err.txt"
    pt_db_json = "c:\\data\\uuid2pt.json"
    for dirname in os.listdir(root_dir):

        print('---------------------')
        print('dirnamem=', dirname)

        # is dirname a MRN?
        if not is_mrn(dirname):
            print(f'dirname({dirname}) is not MRN. So, skipping...')
            continue
        
        # new id
        id = dirname
        new_id = str(uuid.uuid4())

        try:
            # rename dir
            pt_dir_src = os.path.join(root_dir, dirname)
            pt_dir_dst = os.path.join(root_dir, new_id)
            print('{0}->{1}'.format(pt_dir_src, pt_dir_dst))
            os.rename(pt_dir_src, pt_dir_dst)

            # update the info file
            info_file = os.path.join(pt_dir_dst, 'info.txt')
            print('updating info file:', info_file)
            pt = param_from_txt(info_file)
            pt['Id'] = new_id
            pt.save_to_txt(info_file)

            # save to pt db
            print('adding pt to db:', pt_db_json)
            pt_db = param_from_json(pt_db_json) if os.path.exists(pt_db_json) else Param()

            # add to db
            pt_db[new_id] = {
                "Id": id
            }
            # save db
            pt_db.save_to_json(pt_db_json)
        except:
            append_line(err_file, dirname)
    print('done')

#
# python dictionary file is not easy to be handled by other program languages
# better to use csv file. so convering .json to two csv files.
#
def uuid2ptjson_to_csv_files():
    pt_db_json = "c:\\data\\uuid2pt.json"
    uuid2pt_csv = "c:\\data\\uuid2pt.txt"
    id2uuid_csv = "c:\\data\\id2uuid.txt"
    db = param_from_json(pt_db_json)
    for uuid in db.keys():
        pt = db[uuid]
        id = pt["Id"]
        append_line(uuid2pt_csv, '{0}={1}'.format(uuid,id))
        append_line(id2uuid_csv, '{0}={1}'.format(id,uuid))
    print('done')
    
uuid2ptjson_to_csv_files()



