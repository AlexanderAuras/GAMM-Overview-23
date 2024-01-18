#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

#SBATCH --array=0-77

#SBATCH --output=/work/ws-tmp/aa609734-GAMM/runs/unet/%A-%a-stdout.log
#SBATCH --error=/work/ws-tmp/aa609734-GAMM/runs/unet/%A-%a-stderr.log

#SBATCH --job-name="UNet"

/cm/shared/omni/apps/miniconda3/bin/activate GAMM
OVERWRITES=$(python gen_overwrites.py $SLURM_ARRAY_TASK_ID)
"/home/aa609734/.conda/envs/GAMM/bin/python" "/home/aa609734/Projects/GAMM-Overview-23/train.py" $OVERWRITES
