from helper import read_key_value_pairs
from os.path import join, exists
from locnetworker2 import LocNetWorker2
from unetworker import UNetWorker
from param import param_from_json
from providers.TrainingJobsDataProvider import TrainingJobsDataProvider
import os
import dict_helper
import time

class WorkerManager2():
    def __init__(self, wid):
        self.wid = wid

    def get_list_of_jobs(self):
        print('get list of jobs')
        return TrainingJobsDataProvider.get_all()

    def get_list_of_jobs_not_completed(self):
        print('get list of jobs, which is not compelted yet')
        return TrainingJobsDataProvider.get_jobs_not_completed()

    def loop(self, sleep_sec):
        print('loop()...')
        i = 0
        while(True):
            print(f'itr={i}')
            self.run()
            print(f'ran a round. sleeping for {sleep_sec} sec ...')
            time.sleep(sleep_sec)
            i = i+1

    def run(self):
        print(f'worker(wid={self.wid}) started.')
        data_dir = './__data'
        jobs_dir = join(data_dir, 'jobs')
        
        print('data_dir=', data_dir)
        print('jobs_dir=', jobs_dir)

        # get list of jobs
        print('getting a list of jobs not completed...')
        data = self.get_list_of_jobs_not_completed()

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
            
            # create a worker
            worker = self.create_worker(job_file)

            # train network
            # worker.train()

            # run test set
            worker.test()

    def create_worker(self, worker_file):

        if not exists(worker_file):
            print(f'worker file not found:{worker_file}')
            return None
        dict = param_from_json(worker_file)
        type = dict['Type']

        if type.strip() == '':
            print(
                'the field [Model] is not found in the worker file [{worker_file}]')
            return None

        if type == 'LocNet3D':
            return LocNetWorker2(worker_file)
        elif type == 'UNet3D':
            return UNetWorker(worker_file)
        else:
            print(f'worker of type [{type}] not found!')
            return None
        

