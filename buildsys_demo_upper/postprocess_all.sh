#!/usr/bin/env bash

source activate tfgpu

list_files=(
            demo_FastGRNN_2class_winlen_256_stride_128
            )

# outname=rerun_bb_tar_winindex_all.sh
# echo 'outname=`echo $0 | sed "s/.sh/.out/g"`' > $outname

for l in "${list_files[@]}"; do
    echo -e "\n\n-------------------- Processing $l ---------------------"
    sh ../hpc_scripts/4_collate_output_splits.sh $l > $l.out

    # python3 ../hpc_scripts/5_compute_best_tvt.py $l.out $outname
    python3 ../hpc_scripts/5_compute_best_tvt.py $l.out
done

# Change pbs data path to slurm
# sed -i 's|'/fs/project/PAS1090/radar/Bumblebee/'|'/scratch/dr2915/Bumblebee/'|g' $outname

# sh ../hpc_scripts/1_split_hyperparam_file.sh $outname 5
# sh ../hpc_scripts/2_create_batch_jobs.sh "rerun_bb_tar_winindex_all" "02:00:00" slurm
