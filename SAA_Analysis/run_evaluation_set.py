from second_stage_SAA import *
import time
import multiprocessing as mp 

instance = 1
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
epsilons = [1500000, 700000]
rl = 0.5
batches = 30
num_Scenarios = 350
start_time = time.time()
for scen in range(1024):
    with mp.Pool(30) as pool:
        pool.starmap(run_Model, [(scen, batch) for batch in range(batches)])