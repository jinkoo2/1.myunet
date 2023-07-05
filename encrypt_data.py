# Open a file: file
from genericpath import isdir
from helper import read_all_text
import uuid
import jwt
import os
from param import param_from_txt

##################################
# secret from a private key file
secret = read_all_text('.\\_secure\\private_key.pem')
print(secret)

######################
# casename
case_name = str(uuid.uuid4())
print('case_name=', uuid.uuid4())

dir = 'c:\\data'

for name in os.listdir(dir):
    
    pt_dir = os.path.join(dir,name);

    # skip if not a directory
    if os.path.isdir(pt_dir) is False:
        continue

    print('pt_dir=', pt_dir)
    pt_info_file = os.path.join(pt_dir, "info.txt")

    pt_param = param_from_txt(pt_info_file)

    pt_param
    obj = {
    "Id": pt_param["Id"],
    "LastName": pt_param["LastName"],
    "FirstName": pt_param["FirstName"]}

    encoded = jwt.encode(obj, secret,algorithm="HS256")
    print(encoded)

    pt_param["jwt"]=encoded
    pt_param["Id"] = ""
    pt_param["LastName"] = ""
    pt_param["FirstName"] = ""

    pt_param.save_to_txt(pt_info_file+".new.txt")
    
    exit(0)

pt ={
    "first_name": "first",
    "last_name" : "last",
    "Id":"12345678910"
}

secret = "secrete"

encoded_jwt = jwt.encode(pt, secret, algorithm="HS256")
pt2 = jwt.decode(encoded_jwt, secret, algorithms=["HS256"])

print(pt2)




print('done')