#!/usr/bin/env bash
########################################################
# Create batch jobs
########################################################
create_batch_jobs()
{
    split_files=`(ls $1*.sh)`
    for file in $split_files
    do
        echo "sbatch --export=filename=$file batch_job.sbatch"
        echo sleep 1
    done
}

########################################################
# Check number of args
########################################################
if [ $# -ne 1 ]
then
    echo "Usage: $0 <split_file_prefix>" > /dev/stderr
    exit 1
fi

if [[ $1 == *"q15"* ]]; then submit_file=3_SUBMIT_q15_batch_jobs.sh
else submit_file=3_SUBMIT_batch_jobs.sh; fi

create_batch_jobs $1 > $submit_file