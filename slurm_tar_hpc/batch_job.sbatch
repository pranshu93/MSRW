#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5GB
#SBATCH --time=4-0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=roy.174@osu.edu

source activate l3embedding-new
echo "FILENAME: " ${filename}
module purge
module load cudnn/8.0v6.0
bash ${filename}
