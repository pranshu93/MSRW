from multiprocessing.dummy import Pool as ThreadPool
import subprocess 
import numpy as np
import itertools
import os

params = {}
params["ne"] = [100]
params["ksize"] = [8,16,32]
params["clip"] = [2,5]
params["levels"] = [2,3,4]
params["nhid"] = [8,16,32]
params["ot"] = [False,True]

file = open('testfile.txt','w')

script_loc = os.path.abspath('tcn_.py')

def my_function(v):
    global script_loc; global params;
    test_sum = 0
    call_str = "python3 " + script_loc
    ctr = 0
    for k in params.keys():
        call_str += " -" + k + " " + str(v[ctr]); ctr += 1;
        
    for i in range(1,6):	        
        res_str = subprocess.getoutput(call_str + " -fn " + str(i))
        file.write(res_str.split()[-1])
        file.write(' ')
        #test_sum += float(res_str.split()[-1])
    #file.write(str(test_sum/5))
    file.write(call_str)
    file.write('\n')
    return 0	

pool = ThreadPool(5)
my_array = np.asarray(list(itertools.product(*params.values()))).tolist()

results = pool.map(my_function, my_array)

#np.save("results.npy",np.concatenate((np.array(my_array),np.array(results).reshape(-1,1)),1))

file.close()
