from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import itertools
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-pref', type=str, help='Prefix of data file (excluding _RNNspectrogram.csv)')
parser.add_argument('-base', type=str, help='Base path of data')
parser.add_argument('-bat', type=str, default='pbs', help='Batch system (pbs/slurm)')

if len(sys.argv)<4:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

# Create output file
out_folder = os.path.join('..', args.bat + '_hpc')
out_file = os.path.join(out_folder, 'keras_' + args.pref + '.sh')

# Hyperparameters
hidden=[32,64, 128]
epochs=[500]
dropout_rate=[0.2,0.5,0.8]
learning_rate=[0.01,0.001]
batch_size=[128,64,32]
input_dim=[32,64, 128, 256]

# Create hyperparam combos
def generate_trainstring(v):
    call_str = "python3 ../Architecture/keras_rnn_tvt.py -hs " + str(
        v[0]) + " -ep " + str(v[1]) + " -dr " + str(v[2]) + " -bs " + str(
        v[3]) + " -w " + str(v[4]) + " -lr " + str(v[5]) \
              + " -base " + args.base + " -pref " + args.pref + " -out $outname"

    return call_str

pool = ThreadPool()
hyperparams = list(itertools.product(hidden,epochs,dropout_rate,batch_size,input_dim,learning_rate))
results = pool.map(generate_trainstring, hyperparams)

# Flatten
# results = [item for sublist in results for item in sublist]

with open(out_file, 'w') as f:
    print('outname=`echo $0 | sed "s/.sh/.out/g"`', file=f)
with open(out_file, 'a') as f:
    print(*results, sep = "\n", file=f)