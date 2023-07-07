from helper import read_key_value_pairs
from os.path import join, exists
from locnetworker import LocNetWorker
from unetworker import UNetWorker
from param import param_from_json
from providers.TrainingJobsDataProvider import TrainingJobsDataProvider
import os
import dict_helper


class WorkerManager():
    def __init__(self, workers_dir):
        self.workers_dir = workers_dir

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
            
            if job['Model'] == 'LocNet3D':
                print('creating a LockNet3D worker')


    def create_worker(self, worker_name):
        worker_file = join(self.workers_dir, "wkr."+worker_name+".json")
        if not exists(worker_file):
            print(f'worker file not found:{worker_file}')
            return None
        dict = param_from_json(worker_file)
        type = dict['type']

        if type.strip() == '':
            print(
                'the field [type] is not found in the worker file [{worker_file}]')
            return None

        if type == 'LocNet':
            return LocNetWorker(worker_file)
        elif type == 'UNet':
            return UNetWorker(worker_file)
        else:
            print(f'worker of type [{type}] not found!')
            return None
