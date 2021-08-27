from second_stage_SAA import *
import multiprocessing as mp 

instance = 0
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
epsilons = [1500000, 700000]
rl = 0.5
batches = 30
N = [32, 64, 96, 128, 160, 192]

for num_Scenarios in N:
    for scen in range(3200):
        with mp.Pool(30) as pool:
            pool.starmap(run_Model, [(num_Scenarios, scen, batch) for batch in range(batches)], chunksize=1)
        pool.close()