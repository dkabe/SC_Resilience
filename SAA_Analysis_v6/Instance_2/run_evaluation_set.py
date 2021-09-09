from second_stage_SAA import *
import time
import multiprocessing as mp 

instance = 1
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
epsilons = [1500000, 700000]
rl = 0.5
batches = range(30)
N = [64, 128, 192, 256, 320, 384, 448]
start_time = time.time()

for num_Scenarios in N:
    for scen in range(10000):
        with mp.Pool(30) as pool:
            pool.starmap(run_Model, [(num_Scenarios, scen, batch) for batch in (batches)], chunksize=1)
        pool.close()