#!/bin/bash
#SBATCH --job-name="ScoreSDE"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=12G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH --chdir=/usr/stud/steiger/dev/score_sde_pytorch
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
export CUDA_HOME=/storage/local/sysdata/cuda-11.0
export MTG_DATASET_PATH=/storage/user/steiger/dataset/mtg-jamendo
srun /usr/stud/steiger/miniconda3/envs/score_sde_pytorch/bin/python main.py --config ./configs/subvp/mtg_ddpm_continuous.py --mode train --workdir /usr/stud/steiger/exp/score_sde_pytorch/mtg-classical
