import requests

# this should be loaded from a config file
from config import get_config

class TrainingJobsDataProvider():
    
    url = get_config()['webservice_api_url']+'/trainingjobs'

    @staticmethod
    def get_all():
        r = requests.get(url = TrainingJobsDataProvider.url)
        data = r.json()
        return data
    
    @staticmethod
    def get_jobs_not_completed():
        r = requests.get(url = TrainingJobsDataProvider.url+'/not_completed')
        data = r.json()
        return data


    @staticmethod
    def get_filtered(filter):
      pass

    @staticmethod
    def get_one(id):
        pass

    @staticmethod
    def add(obj):
        pass

    @staticmethod
    def delete(id):
        pass

    @staticmethod
    def update(job):
        #print(job)
        url = TrainingJobsDataProvider.url+'/'+job['_id']
        #print(url)
        r = requests.put(url = url, json=job)
        data = r.json()
        return data
    
    @staticmethod
    def upload_file(job_id, file_path):
        #print('job_id=', job_id)
        #print('file_path=', file_path)
        url = TrainingJobsDataProvider.url+'/upload_file/'+job_id
        #print(url)
        
        files = {'file': open(file_path, 'rb')}
       
        try:
            response = requests.post(url = url, files=files)
            if response.status_code != 200:
                print(f"Error uploading file. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
    
    # @staticmethod
    # def update_properties(job_id, properties):
    #     #print(job)
    #     url = TrainingJobsDataProvider.url+'/'+job_id
    #     #print(url)
    #     r = requests.put(url = url, json=properties)
    #     data = r.json()
    #     return data
    
