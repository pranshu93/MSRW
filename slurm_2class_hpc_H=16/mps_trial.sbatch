#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=FAIL
#PRINCE PRINCE_GPU_MPS=YES

source activate tfgpu
module purge
module load cudnn/9.0v7.3.0.29

outname=`echo $0 | sed "s/.sh/.out/g"`

python3 ../Architecture/dynamic_rnn_tvt.py -ggnl sigmoid -gunl tanh -ur 0.25 -wr 0.25 -w 16 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 0 -type embedding \
-q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_256_winindex_all/HumanVsNonhuman_48_16_Winlen256_gN=sigmoid_uN=tanh_embedding_H=16_k=14_bs=64_ep=10_it=5_rnd=10 -out $outname

python3 ../Architecture/dynamic_rnn_tvt.py -ggnl sigmoid -gunl tanh -ur 0.25 -wr 0.75 -w 16 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 0 -type embedding \
-q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_256_winindex_all/HumanVsNonhuman_48_16_Winlen256_gN=sigmoid_uN=tanh_embedding_H=16_k=14_bs=64_ep=10_it=5_rnd=10 -out $outname

python3 ../Architecture/dynamic_rnn_tvt.py -ggnl sigmoid -gunl tanh -ur 0.75 -wr 0.25 -w 16 -sp 0.5 -lr 0.005 -bs 64 -hs 16 -ot 0 -type embedding \
-q15 False -base /scratch/dr2915/Bumblebee/bb_3class_winlen_256_winindex_all/HumanVsNonhuman_48_16_Winlen256_gN=sigmoid_uN=tanh_embedding_H=16_k=14_bs=64_ep=10_it=5_rnd=10 -out $outname

wait
