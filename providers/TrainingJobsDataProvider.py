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
    
    # @staticmethod
    # def update_properties(job_id, properties):
    #     #print(job)
    #     url = TrainingJobsDataProvider.url+'/'+job_id
    #     #print(url)
    #     r = requests.put(url = url, json=properties)
    #     data = r.json()
    #     return data
    
