import sys
import os

######################### ONLY MODIFY THESE VALUES #########################
# Script prefix
prefix='3class_winlen_256'

# Number of splits of hyperparam file
num_splits='54'

# Base path of data
base='/scratch/dr2915/Bumblebee/bb_3class_winlen_256_winindex_all/'

# Batch system
bat_sys='slurm'

# Running time
walltime='1-0'

######################### KEEP THE REST INTACT #########################
# Folder where jobs are saved
jobfolder = os.path.join('..', bat_sys + '_hpc')

#Init args
init_argv=sys.argv

# Enter hpc_scripts folder
os.chdir('../hpc_scripts')

# Generate gridsearch
print('###### hpc_scripts/gridsearch #####')
sys.argv=init_argv+['-pref', prefix, '-bat', bat_sys, '-base', base]
import hpc_scripts.gridsearch_keras_tvt_0

# Split hyperparam file
print('###### hpc_scripts/split_hyp_wrapper #####')
sys.argv=init_argv+[os.path.join(jobfolder,'keras_' + prefix + '.sh'),num_splits]
import hpc_scripts.split_hyp_wrapper_1

# Create batch job
print('###### hpc_scripts/create_batch_wrapper #####')
sys.argv=init_argv+[os.path.join(jobfolder,'keras_' + prefix + '_'),walltime,bat_sys]
import hpc_scripts.create_batch_wrapper_2

# Submit
print("\nNow submit " + bat_sys + "_hpc/3_SUBMIT_keras_" + prefix + "_jobs.sh on server")