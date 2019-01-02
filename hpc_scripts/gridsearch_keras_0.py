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
hidden=[64,100,200,256]
epochs=[500]
dropout_rate=[0.8,0.9]
learning_rate=[0.01,0.001]
batch_size=[100,64,32]
optimizer=[0, 1]
input_dim=[32,64, 128, 256, 512]
classifier=[0, 1]

# Create hyperparam combos
def generate_trainstring(v):
    call_str = "python3 ../Architecture/keras_rnn.py -hs " + str(
        v[0]) + " -ep " + str(v[1]) + " -dr " + str(v[2]) + " -lr " + str(v[3]) + " -bs " + str(
        v[4]) + " -ot " + str(v[5]) + " -w " + str(v[6]) + " -ct " + str(v[7])\
              + " -base " + args.base + " -pref " + args.pref + " -out $outname"

    return call_str

pool = ThreadPool()
hyperparams = list(itertools.product(hidden,epochs,dropout_rate,learning_rate,batch_size,optimizer,input_dim,classifier))
results = pool.map(generate_trainstring, hyperparams)

# Flatten
# results = [item for sublist in results for item in sublist]

with open(out_file, 'w') as f:
    print('outname=`echo $0 | sed "s/.sh/.out/g"`', file=f)
with open(out_file, 'a') as f:
    print(*results, sep = "\n", file=f)