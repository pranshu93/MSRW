sbatch -t 1-0 --export=filename=../slurm_2class_hpc_H=16/embedding_256_1_spl.sh batch_job.sbatch
sleep 1m
sbatch -t 1-0 --export=filename=../slurm_2class_hpc_H=16/embedding_256_2_spl.sh batch_job.sbatch
