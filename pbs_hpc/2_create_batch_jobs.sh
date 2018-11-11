#!/usr/bin/env bash
########################################################
# Create batch jobs
########################################################
create_batch_jobs()
{
    split_files=`(ls $1*_spl.sh)`

    for file in $split_files
    do
        echo "qsub -l walltime=$2 -v filename=$file batch_job.pbs"
        echo sleep 1
    done
}

########################################################
# Check number of args
########################################################
if [ $# -ne 2 ]
then
    echo "Usage: $0 <split_file_prefix> <job_time_limit>" > /dev/stderr
    exit 1
fi

submit_file=3_SUBMIT_$1jobs.sh

create_batch_jobs $1 $2 > $submit_file
