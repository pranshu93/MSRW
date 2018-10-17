#!/usr/bin/env bash
########################################################
# Check number of args
########################################################
if [ $# -ne 1 ]
then
    echo "Usage: $0 <split_file_prefix>" > /dev/stderr
    exit 1
fi

########################################################
# Create batch jobs
########################################################
split_files=`(ls $1*.sh)`

for file in $split_files
do
	echo "qsub -v  filename=$file batch_job.pbs"
	echo sleep 1
done
