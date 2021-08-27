#!/bin/bash
#SBATCH --account=def-wahab
#SBATCH --time=0-03:00        # time limit (D-HH:MM)
#SBATCH --cpus-per-task=3     # number of CPUs (threads) to use
#SBATCH --mem-per-cpu=6000M   # memory per CPU (in MB)
module load nixpkgs/16.09
module load gurobi/9.0.1
source ~/env_gurobi/bin/activate
python /home/dkabe/Model_brainstorming/SAA_Analysis_v2/run_SAA.py