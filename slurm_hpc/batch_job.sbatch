#!/usr/bin/env bash

#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=roy.174@osu.edu

source activate tfgpu
echo "FILENAME: " ${filename}
module purge
module load cudnn/8.0v6.0
bash ${filename}
