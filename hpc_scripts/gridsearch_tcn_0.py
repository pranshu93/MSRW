from multiprocessing.dummy import Pool as ThreadPool
import sys
import argparse
import subprocess
import numpy as np
import itertools
import os

parser = argparse.ArgumentParser()
parser.add_argument('-type', type=str, default='tar', help='tar/act/any other prefix')
parser.add_argument('-base', type=str, help='Base path of data')
parser.add_argument('-bat', type=str, default='pbs', help='Batch system (pbs/slurm)')

if len(sys.argv)<2:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

params = {}
params["ne"] = [100]
params["ksize"] = [8, 16, 32]
params["clip"] = [2, 5]
params["levels"] = [1, 2, 3, 4]
params["nhid"] = [8, 16, 32]
params["ot"] = [False, True]

out_folder = os.path.join('..',args.bat+'_hpc')
out_file = os.path.join('..', args.bat+'_hpc', 'tcn_' + args.type + '.sh')

#file = open('testfile.txt', 'w')
#file.write('outname=`echo $0 | sed "s/.sh/.out/g"`')

#script_loc = os.path.abspath('tcn_.py')
script_loc = '../TCN/tcn_.py'

def my_function(v):
    global script_loc
    global params

    call_str = "python3 " + script_loc
    ctr = 0
    for k in params.keys():
        call_str += " -" + k + " " + str(v[ctr])
        ctr += 1

    res_str = []
    for i in range(1, 6):
        #res_str = subprocess.getoutput(call_str + " -fn " + str(i))
        res_str.append(call_str + " -fn " + str(i) + ' -base ' + args.base + ' -type ' + args.type + ' -out $outname')
        #file.write(res_str.split()[-1])
        #file.write(' ')
        # test_sum += float(res_str.split()[-1])

    return res_str


pool = ThreadPool(5)
my_array = np.asarray(list(itertools.product(*params.values()))).tolist()

results = pool.map(my_function, my_array)

# Flatten
results = [item for sublist in results for item in sublist]

with open(out_file, 'w') as f:
    print('outname=`echo $0 | sed "s/.sh/.out/g"`', file=f)
with open(out_file, 'a') as f:
    print(*results, sep = "\n", file=f)

# np.save("results.npy",np.concatenate((np.array(my_array),np.array(results).reshape(-1,1)),1))
#file.close()
