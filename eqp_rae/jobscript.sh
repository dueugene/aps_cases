#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=12:00:00
#SBATCH --output=rae_subsonic_adaptive_greedy_eqp_%j.txt
#SBATCH --job-name rae_subsonic_adaptive_greedy_eqp
#SBATCH --mail-type=FAIL
#SBATCH -A ctb-myano

cd $SLURM_SUBMIT_DIR

source ~/aps_aux/setup_env_scinet.sh

mpirun ./rae > log.log
