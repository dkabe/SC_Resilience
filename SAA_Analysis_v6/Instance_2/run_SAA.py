from SAA import *
import multiprocessing as mp 
instance = 1
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
epsilons = [1500000, 700000]
rl = 0.5
batches = range(30)
#N = [50]
N = [64, 128, 192, 256, 320, 384, 448]
N = [192, 256, 320, 384]


#for batch in range(batches):

 #   run_Model(instance, rl, num_Scenarios, MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], batch)

for num_scenarios in N:
    with mp.Pool(30) as pool:
        pool.starmap(run_Model, [(num_scenarios, batch) for batch in batches], chunksize=1)
