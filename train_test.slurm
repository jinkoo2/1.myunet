#!/bin/bash

#SBATCH --job-name=train_test
#SBATCH --output=train_test.py.log
#SBATCH --ntasks-per-node=28
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH -p gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jinkoo.kim@stonybrook.edu

module load pytorch

python /gpfs/home/jinkokim/1.myunet/train_test_gpu.py