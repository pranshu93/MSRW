from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import itertools
import ast
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-type', type=str, default='tar', help='tar/act/any other prefix')
parser.add_argument('-base', type=str, help='Base path of data')
parser.add_argument('-bat', type=str, default='pbs', help='Batch system (pbs/slurm)')
parser.add_argument('-big', type=ast.literal_eval, default=False, help='Is this a big gridsearch?')
parser.add_argument('-q15', type=ast.literal_eval, default=False, help='Is this a Q15 gridsearch?')

if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

if args.q15:
    p12 = ['quantSigm', 'quantTanh']  # ['relu','quantSigm','quantTanh'] #['quantSigm']
    p13 = ['quantTanh', 'quantSigm']  # ['relu','quantSigm','quantTanh'] #['quantTanh']
else:
    p12 = ['sigmoid', 'tanh']
    p13 = ['tanh', 'sigmoid']

if args.big:
    p14 = [0.25, 0.75]  # [0.25,0.5,0.75,1] #[1]
    p15 = [0.25, 0.75]  # [0.25,0.5,0.75,1] #[1]
    p2 = [32, 64, 96, 128]  # [32]
    p3 = [0.25, 0.5, 0.75, 1]  # [0.5]
    p4 = [0.005]  # [0.005,0.01]
    p5 = [32, 64, 96, 128]  # [128]
    p6 = [16, 48]  # [16,32,48,64] #[16]
    p7 = [0]  # [1]
    # p8 = [512,768,1024] #[768]
else:
    p14 = [0.25, 0.75]
    p15 = [0.25, 0.75]
    p2 = [32]
    p3 = [0.5]
    p4 = [0.005]
    p5 = [128]
    p6 = [16, 32]
    p7 = [0, 1]
    # p8 = [768] # no longer used

out_folder = os.path.join('..', args.bat + '_hpc')
out_suffix = ''
if args.big:
    out_suffix += '_big'
if args.q15:
    out_suffix += '_q15'
out_file = os.path.join('..', args.bat + '_hpc', args.type + out_suffix + '.sh')


def generate_trainstring(v):
    res_str = "python3 ../Architecture/dynamic_rnn.py -ggnl " + str(
        v[0]) + " -gunl " + str(v[1]) + " -ur " + str(v[2]) + " -wr " + str(v[3]) + " -w " + str(
        v[4]) + " -sp " + str(v[5]) + " -lr " + str(v[6]) + " -bs " + str(v[7]) + " -hs " + str(
        v[8]) + " -ot " + str(v[9]) + " -type " + args.type + " -q15 " + str(
        args.q15) + " -base " + args.base + " -out $outname"

    return res_str

pool = ThreadPool()
hyperparams = np.asarray(list(itertools.product(p12, p13, p14, p15, p2, p3, p4, p5, p6, p7))).tolist()
results = pool.map(generate_trainstring, hyperparams)

# Flatten
results = [item for sublist in results for item in sublist]

with open(out_file, 'w') as f:
    print('outname=`echo $0 | sed "s/.sh/.out/g"`', file=f)
with open(out_file, 'a') as f:
    print(*results, sep="\n", file=f)
