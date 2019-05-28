#!/usr/bin/env bash

source activate tfgpu

list_files=(
            bb_tar_winlen_256_winindex_all
            bb_tar_winlen_384_winindex_all
            bb_tar_winlen_512_winindex_all
            bb_tar_winlen_640_winindex_all
            bb_tar_winlen_768_winindex_all
            )

for l in ${list_files[@]}; do
    echo -e "\n\n-------------------- Processing $l ---------------------"
    sh ../hpc_scripts/4_collate_output_splits.sh $l > $l.out

    python3 ../hpc_scripts/5_compute_best_tvt.py $l.out
done
