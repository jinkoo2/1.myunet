import requests

config={
    'webservice_url': 'http://roweb3.uhmc.sbuh.stonybrook.edu:3000/api'
}

class TrainingJobsDataProvider():
    
    url = config['webservice_url']+'/trainingjobs'

    @staticmethod
    def get_all():
        r = requests.get(url = TrainingJobsDataProvider.url)
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
    def update(obj):
        pass
