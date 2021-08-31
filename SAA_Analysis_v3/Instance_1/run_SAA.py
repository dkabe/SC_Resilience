from SAA import *
import multiprocessing as mp 
instance = 0
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
epsilons = [1500000, 700000]
rl = 0.5
batches = [0,21]
N = [32, 64, 96, 128, 160, 192]
N = [256]


#for batch in range(batches):

 #   run_Model(instance, rl, num_Scenarios, MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], batch)

for num_scenarios in N:
    with mp.Pool(2) as pool:
        pool.starmap(run_Model, [(num_scenarios, batch) for batch in batches], chunksize=1)
