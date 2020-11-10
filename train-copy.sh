#!/bin/bash

#The name of the job is train small data
#SBATCH -J y

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is 8 days
#SBATCH -t 192:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=x@ut.ee

#SBATCH --mem=150GB

#SBATCH --partition=gpu

#SBATCH --gres=gpu:tesla:1

module load python-3.7.1

source activate env-name


python3 yolov3/train.py --cfg yolov3/cfg/yolov3-spp.cfg --data yolov3/data/x.data --name x-model
