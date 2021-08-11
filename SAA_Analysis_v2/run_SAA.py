from SAA import *
import multiprocessing as mp 
instance = 1
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
epsilons = [1500000, 700000]
rl = 0.5
batches = [2,3,4,5,6,10,16,18,19,20,21,22,23,24,27,28]
#N = [100, 200, 300, 400, 500]
N = [500]

#for batch in range(batches):

 #   run_Model(instance, rl, num_Scenarios, MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], batch)

for num_scenarios in N:
    with mp.Pool(30) as pool:
        pool.starmap(run_Model, [(num_scenarios, batch) for batch in (batches)])
