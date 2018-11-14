import sys
import os

print(os.getcwd())
os.chdir('hpc_scripts')

# Script prefix
prefix='m_30_n_96_sigpad'

# Folder where jobs are saved
jobfolder = '../pbs_hpc/'

# Base path of data
base='/fs/project/PAS1090/radar/Austere/Bora_New_Detector/Bora_new_det_aus_M_30_N_96_win_res_last_w_padded_with_signal_lookahead'

#Init args
init_argv=sys.argv

# Prepare data
print('###### Scripts/processing_data #####')
sys.argv=init_argv+['-type', prefix, '-base', base, '-hum','austere_386_human','-nhum','austere_310_cow']
import Scripts.processing_data

# Generate gridsearch
print('###### hpc_scripts/gridsearch #####')
sys.argv=init_argv+['-type', prefix, '-bat', 'pbs', '-base', base]
import hpc_scripts.gridsearch_0

# Split hyperparam file
print('###### hpc_scripts/split_hyp_wrapper #####')
sys.argv=init_argv+[jobfolder+prefix+'.sh','32']
import hpc_scripts.split_hyp_wrapper_1

# Create batch job
print('###### hpc_scripts/create_batch_wrapper #####')
sys.argv=init_argv+[jobfolder+prefix+'_','3:00:00','pbs']
import hpc_scripts.create_batch_wrapper_2

# Submit
print("\nNow submit pbs_hpc/3_SUBMIT_"+prefix+"_jobs.sh on PBS server")