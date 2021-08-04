#!/bin/bash
#SBATCH --account=def-wahab
#SBATCH --time=0-00:02        # time limit (D-HH:MM)
#SBATCH --cpus-per-task=1     # number of CPUs (threads) to use
#SBATCH --mem-per-cpu=256M   # memory per CPU (in MB)

python /home/dkabe/Model_brainstorming/test_file.py