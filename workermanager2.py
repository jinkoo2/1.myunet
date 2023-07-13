from helper import read_key_value_pairs
from os.path import join, exists
from locnetworker2 import LocNetWorker2
from unetworker import UNetWorker
from param import param_from_json
from providers.TrainingJobsDataProvider import TrainingJobsDataProvider
import os
import dict_helper

class WorkerManager2():
    def __init__(self, config):
        self.config = config

    def get_list_of_jobs(self):
        print('get list of jobs')
        
        return TrainingJobsDataProvider.get_all()

    def run(self):
        
        data_dir = './__data'
        jobs_dir = join(data_dir, 'jobs')
        data = self.get_list_of_jobs()
        
        print('N=', data['totalCount'])
        for job in data['list']:
            print(job['_id'], job['Name'], job['Model'], job['Status'])
            # create a job dir if not exist
            job_dir =  join(jobs_dir, job['_id'])
            if not exists(job_dir):
                print('creating job dir... ', job_dir)
                os.makedirs(job_dir)

            # save the job (worker_file) as json
            job_file = join(job_dir, 'worker_job.json')
            print('saving job... ', job_file)
            dict_helper.save_to_json(job, job_file)
            
            worker = self.create_worker(job_file)

            worker.train()

    def create_worker(self, worker_file):

        if not exists(worker_file):
            print(f'worker file not found:{worker_file}')
            return None
        dict = param_from_json(worker_file)
        model = dict['Model']

        if model.strip() == '':
            print(
                'the field [Model] is not found in the worker file [{worker_file}]')
            return None

        if model == 'LocNet3D':
            return LocNetWorker2(worker_file)
        elif model == 'UNet3D':
            return UNetWorker(worker_file)
        else:
            print(f'worker of type [{model}] not found!')
            return None
