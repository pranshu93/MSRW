from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import itertools
import argparse
import sys

p12 = ['quantSigm','quantTanh'] #['relu','quantSigm','quantTanh'] #['quantSigm']
p13 = ['quantTanh','quantSigm'] #['relu','quantSigm','quantTanh'] #['quantTanh']
p14 = [0.25,0.75] #[0.25,0.5,0.75,1] #[1]
p15 = [0.25,0.75] #[0.25,0.5,0.75,1] #[1]
p2 =  [32,64,96,128] #[32]
p3 =  [0.25,0.5,0.75,1] #[0.5]
p4 = [0.005] #[0.005,0.01]
p5 = [32,64,96,128] #[128]
p6 = [16,48] #[16,32,48,64] #[16] 
p7 = [0] #[1] 
p8 = [512,768,1024] #[768]

parser = argparse.ArgumentParser()
parser.add_argument('-type', type=str, default='tar', help='tar/act')

if len(sys.argv)==1:
    parser.print_help()
    exit(1)

args = parser.parse_args()

def generate_trainstring(v):
    call_str = "python3 ../Architecture/dynamic_rnn.py -ggnl " + str(
        v[0]) + " -gunl " + str(v[1]) + " -ur " + str(v[2]) + " -wr " + str(v[3]) + " -w " + str(
        v[4]) + " -sp " + str(v[5]) + " -lr " + str(v[6]) + " -bs " + str(v[7]) + " -hs " + str(
        v[8]) + " -ot " + str(v[9]) + " -ml " + str(v[10])

    res_str=[]
    for i in range(1,6):
        res_str.append(call_str + " -fn " + str(i) + " -type " + args.type + " -bat slurm -out $outname")

    return res_str

pool = ThreadPool()
hyperparams = np.asarray(list(itertools.product(p12,p13,p14,p15,p2,p3,p4,p5,p6,p7,p8))).tolist()
results = pool.map(generate_trainstring, hyperparams)

# Flatten
results = [item for sublist in results for item in sublist]

with open(args.type + '_big.sh', 'w') as f:
    print('outname=`echo $0 | sed "s/.sh/.out/g"`', file=f)
with open(args.type + '_big.sh', 'a') as f:
    print(*results, sep = "\n", file=f)
