import requests
import os

# this should be loaded from a config file
from config import get_config
from param import Param
class StructuresDataProvider():
    
    config = get_config()
    api_url = config['webservice_api_url']+'/structures'
    data_url = config['webservice_data_url']

    @staticmethod
    def get_all():
        r = requests.get(url = StructuresDataProvider.api_url)
        data = r.json()
        return data
    
    @staticmethod
    def get_filtered(filter):
      pass

    @staticmethod
    def get_one(id):
        r = requests.get(url = StructuresDataProvider.api_url+'/'+id)
        data = r.json()
        return data

    @staticmethod
    def add(obj):
        pass

    @staticmethod
    def delete(id):
        pass

    @staticmethod
    def update(obj):
        pass

    @staticmethod
    def download_file(url, out_file_path):
        res = requests.get(url)
        # Check if the request was successful (status code 200)
        if res.status_code == 200:
            with open(out_file_path, 'wb') as file:
                file.write(res.content)
            #print('File downloaded successfully.')
        else:
            msg = f'''Failed to download the file: {res.status_code}
                        url={url} 
                        out_file_path={out_file_path}'''
            print(msg)
            raise Exception(msg)

    @staticmethod
    def download_files(id):
        
        print('download_files for '+id)

        # get structure from db
        s = StructuresDataProvider.get_one(id)

        # download structure files (mhd, zraw, info)
        str_mhd = StructuresDataProvider.download_structure_files(s)
        
        # download image files (mhd, zraw, info)
        img_mhd = StructuresDataProvider.download_image_files(s)

        return {"structure_mhd": str_mhd, "image_mhd": img_mhd}
    
    def download_structure_files(structure_object):
        
        s = structure_object

        # download to local folder
        dst_dir = StructuresDataProvider.config['structures_root_dir']+'/'+s['_id']
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)    

        #mhd
        src_url = StructuresDataProvider.data_url+'/'+s['str_file_path']
        dst_file = dst_dir + '/img.mhd'
        if not os.path.exists(dst_file):
            StructuresDataProvider.download_file(src_url, dst_file)
        mhd = dst_file

        #zraw
        src_url = src_url.replace('.mhd', '.zraw')
        dst_file = dst_file.replace('.mhd', '.zraw')
        if not os.path.exists(dst_file):
            StructuresDataProvider.download_file(src_url, dst_file)
        #info
        src_url = src_url.replace('.zraw', '.info')
        dst_file = dst_file.replace('.zraw', '.info')
        if not os.path.exists(dst_file):
            StructuresDataProvider.download_file(src_url, dst_file)

        return mhd
    
    def download_image_files(structure_object):
        
        s = structure_object
            
        # download to local folder
        dst_dir = StructuresDataProvider.config['images_root_dir']+'/'+s['sset']['img']
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)    
        
        #mhd
        src_url = StructuresDataProvider.data_url+'/'+s['img_file_path']
        dst_file = dst_dir + '/img.mhd'
        if not os.path.exists(dst_file):
            StructuresDataProvider.download_file(src_url, dst_file)
        mhd = dst_file

        #zraw
        src_url = src_url.replace('.mhd', '.zraw')
        dst_file = dst_file.replace('.mhd', '.zraw')
        if not os.path.exists(dst_file):
            StructuresDataProvider.download_file(src_url, dst_file)
        #info
        src_url = src_url.replace('img.zraw', 'info.txt')
        dst_file = dst_file.replace('.zraw', '.info')
        if not os.path.exists(dst_file):
            StructuresDataProvider.download_file(src_url, dst_file)

        return mhd