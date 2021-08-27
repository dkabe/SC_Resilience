from SAA import *
import multiprocessing as mp 
instance = 1
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
epsilons = [1500000, 700000]
rl = 0.5
batches = [5, 12, 25]

#N = [100, 200, 300, 400, 500]
N = [1024]

#for batch in range(batches):

 #   run_Model(instance, rl, num_Scenarios, MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], batch)

for num_scenarios in N:
    with mp.Pool(3) as pool:
        pool.starmap(run_Model, [(num_scenarios, batch) for batch in batches])
