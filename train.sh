#!/bin/bash
#SBATCH -J dc-gan
#SBATCH -p cuda
#SBATCH -o out/dc-gan-%j.out
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

source activate yolov3
python train.py