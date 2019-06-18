#!/usr/bin/env bash

source activate tfgpu

list_dir=(
    ../slurm_2class_hpc_H=16
    ../slurm_2class_hpc_H=32
    ../slurm_2class_hpc_H=64
    )

list_files=(
            embedding_256
            embedding_384
            embedding_512
            embedding_640
            embedding_768
            )

for dir in ${list_dir[@]}; do
    h="$(echo $dir | cut -d'=' -f2)"
    
    # Collate outputs first
    for l in ${list_files[@]}; do
	cd $dir
	sh ../hpc_scripts/4_collate_output_splits.sh $l > H_${h}_${l}.out
    done
done

for dir in ${list_dir[@]}; do
    h="$(echo $dir | cut -d'=' -f2)"

    echo -e "\n\n\t\t-------------------- Hidden size = $h ---------------------"
    for l in ${list_files[@]}; do
	cd $dir
        echo -e "\n\t----- Processing $l -----"
        python3 ../hpc_scripts/5_compute_best_tvt_hiddenfiltered.py H_${h}_${l}.out $h
    done
done
