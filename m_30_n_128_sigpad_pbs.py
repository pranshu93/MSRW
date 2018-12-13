import sys
import os

######################### ONLY MODIFY THESE VALUES #########################
# Script prefix
prefix='m_30_n_128_sigpad'

# Number of splits of hyperparam file
num_splits='32'

# Base path of data
base='/fs/project/PAS1090/radar/Austere/Bora_New_Detector/Bora_new_det_aust_th_22_bgr_19_M_30_N_128_window_res_lookahead_last_wind_padded'

# Batch system
bat_sys='pbs'

# Human and nonhuman folders
hum_fold='austere_384_human'
nhum_fold='austere_304_cow'

# Running time
walltime='3:00:00'

######################### KEEP THE REST INTACT #########################
# Folder where jobs are saved
jobfolder = '../'+ bat_sys +'_hpc/'

#Init args
init_argv=sys.argv

# Enter hpc_scripts folder
os.chdir('hpc_scripts')

# Prepare data
print('###### Scripts/processing_data #####')
sys.argv=init_argv+['-type', prefix, '-base', base, '-hum', hum_fold, '-nhum', nhum_fold]
import Scripts.processing_data

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
print("\nNow submit " + bat_sys + "_hpc/3_SUBMIT_"+prefix+"_jobs.sh on server")