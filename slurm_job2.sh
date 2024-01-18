#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

#SBATCH --output=/work/ws-tmp/aa609734-GAMM/runs/tmp/%j-stdout.log
#SBATCH --error=/work/ws-tmp/aa609734-GAMM/runs/tmp/%j-stderr.log

#SBATCH --job-name="Tiramisu"

/cm/shared/omni/apps/miniconda3/bin/activate GAMM
"/home/aa609734/.conda/envs/GAMM/bin/python" "/home/aa609734/Projects/GAMM-Overview-23/train2.py" --epochs=700 --lr=1e-5 --ckpt=_WINNER_tense-distortion --batch_acc=2
