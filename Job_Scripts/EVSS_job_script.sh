#!/bin/bash
#SBATCH --account=def-wahab
#SBATCH --time=0-01:00        # time limit (D-HH:MM)
#SBATCH --cpus-per-task=40     # number of CPUs (threads) to use
#SBATCH --mem-per-cpu=256M   # memory per CPU (in MB)
module load nixpkgs/16.09
module load gurobi/9.0.1
source ~/env_gurobi/bin/activate
python /home/dkabe/SC_Resilience/EVSS/run_EVSS.py