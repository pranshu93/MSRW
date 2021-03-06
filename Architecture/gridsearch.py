from multiprocessing.dummy import Pool as ThreadPool
import subprocess 
import numpy as np
import itertools
import sys
import tensorflow as tf
import os
from rnn import FastRNNCell,FastGRNNCell

#p1 = [1] #[0,1]
#p11 = [] #['relu','quantSigm','quantTanh']
p12 = ['quantSigm','quantTanh'] #['relu','quantSigm','quantTanh'] #['quantSigm']
p13 = ['quantTanh','quantSigm'] #['relu','quantSigm','quantTanh'] #['quantTanh']
p14 = [0.25,0.75] #[0.25,0.5,0.75,1] #[1]
p15 = [0.25,0.75] #[0.25,0.5,0.75,1] #[1]
p2 =  [32,64,96,128] #[32]
p3 =  [0.25,0.5,0.75,1] #[0.5]
p4 = [0.005] #[0.005,0.01]
p5 = [32,64,96,128] #[128]
p6 = [16,48] #[16,32,48,64] #[16] 
p7 = [0,1] #[1] 
p8 = [512,640,768,896,1024] #[768]

file = open('testfile.txt','w')

def my_function(v):
    test_sum = 0; try_sum = 0;
    call_str = "python3 /home/cse/phd/anz178419/MSRW/Architecture/dynamic_rnn.py -ggnl " + str(v[0]) + " -gunl " + str(v[1]) + " -ur " + str(v[2]) + " -wr " + str(v[3]) + " -w " + str(v[4]) + " -sp " + str(v[5]) + " -lr " + str(v[6]) + " -bs " + str(v[7]) + " -hs " + str(v[8]) + " -ot " + str(v[9]) + " -ml " + str(v[10])
    for i in range(1,6):	        
        res_str = subprocess.getoutput(call_str + " -fn " + str(i))
        print(res_str)
        #print(res_str.split()[res_str.split().__len__()-2:],sep=" ")
        #print(v)
        test_sum += float(res_str.split()[res_str.split().__len__()-2:][0]); try_sum += float(res_str.split()[res_str.split().__len__()-2:][1])
    file.write(str(test_sum/5))
    file.write(' ')
    file.write(str(try_sum/5))
    file.write(' ')
    file.write(call_str)
    file.write('\n')
    #print(test_sum/5,try_sum/5,v)
    return (test_sum/5,try_sum/5)	

pool = ThreadPool(96)
#my_array = np.asarray(list(itertools.product(p0,p1,p2,p3,p4,p5,p6))).tolist()
#my_array = np.asarray(list(itertools.product(p2,p4,p5,p6))).tolist()
#my_array = np.asarray(list(itertools.product(p0,p1))).tolist()
my_array = np.asarray(list(itertools.product(p12,p13,p14,p15,p2,p3,p4,p5,p6,p7,p8))).tolist()

results = pool.map(my_function, my_array)

np.save("results.npy",np.concatenate((np.array(my_array),np.array(results).reshape(-1,2)),1))

file.close()
