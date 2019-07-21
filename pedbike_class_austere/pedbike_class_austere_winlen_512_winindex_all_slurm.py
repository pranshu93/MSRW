import sys
import os

sys.path.append('../../')

######################### ONLY MODIFY THESE VALUES #########################
# Winlen
winlen=512

# Script prefix
prefix='pedbike_class_winlen_' + str(winlen) + '_winindex_all'

# Number of splits of hyperparam file
num_splits='32'

# Base path of data
base='/scratch/dr2915/austere/classification_data_windowed/winlen_' \
     + str(winlen) + '_winindex_all/pedbike_class_winlen_' + str(winlen) + '_winindex_all'

# Batch system
bat_sys='slurm'

# Human and nonhuman folders
#hum_fold='austere_404_human'
#nhum_fold='Bike_485_radial'

# Running time
walltime='1-0'

######################### KEEP THE REST INTACT #########################
# Enter hpc_scripts folder
os.chdir('../hpc_scripts')

# Folder where jobs are saved
jobfolder = '../'+ bat_sys +'_hpc/'

#Init args
init_argv=sys.argv

# Prepare data
# print('###### Scripts/create_train_val_test_split #####')
# sys.argv=init_argv+['-type', prefix, '-base', base]
# import Scripts.create_train_val_test_split

# Generate gridsearch
print('###### hpc_scripts/gridsearch #####')
sys.argv=init_argv+['-type', prefix, '-bat', bat_sys, '-base', base]
import hpc_scripts.gridsearch_0

# Split hyperparam file
print('###### hpc_scripts/split_hyp_wrapper #####')
sys.argv=init_argv+[jobfolder+prefix+'.sh',num_splits]
import hpc_scripts.split_hyp_wrapper_1

# Create batch job
print('###### hpc_scripts/create_batch_wrapper #####')
sys.argv=init_argv+[jobfolder+prefix+'_',walltime,bat_sys]
import hpc_scripts.create_batch_wrapper_2

# Submit
print("\nNow submit " + jobfolder + "3_SUBMIT_"+prefix+"_jobs.sh on server")