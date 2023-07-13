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
