#!/bin/bash
#SBATCH -J gan-age
#SBATCH -p cuda
#SBATCH -o out/gane-age-cr-%j.out
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mail-user=itdfh@umsystem.edu
#SBATCH --mail-type=ALL

source activate yolov3
python train_cond.py