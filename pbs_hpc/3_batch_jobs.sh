qsub -v  filename=train_hyperparams_01.sh batch_job.pbs
sleep 1
qsub -v  filename=train_hyperparams_02.sh batch_job.pbs
sleep 1
qsub -v  filename=train_hyperparams_03.sh batch_job.pbs
sleep 1
qsub -v  filename=train_hyperparams_04.sh batch_job.pbs
sleep 1
qsub -v  filename=train_hyperparams_05.sh batch_job.pbs
sleep 1
