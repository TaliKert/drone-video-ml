#!/bin/bash
#BATCH -J train-yolo-tiny
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=25000
#SBATCH -t 02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

module load cuda/10.2.89-2fkd

source ../torchenv/bin/activate

## Run the following block once to install nvidia apex utilities in the virtual environment
#module load gcc-5.2.0
#cd apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#cd ..

python train.py --data data/person-coco.data --cfg cfg/yolov3-tiny-1cls.cfg --weights=weights/last.pt --single-cls --batch-size 42 --epochs 20