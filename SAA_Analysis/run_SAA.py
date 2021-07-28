from SAA import *
import time
instance = 5
MPs = [2, 3, 4, 6, 6, 6]
DCs = [3, 4, 6, 8, 4, 4]
MZs = [1, 2, 3, 5, 29, 29]
epsilons = [200, 400, 600, 800, 1500000, 700000]
rl = 0.5
batches = 30
num_Scenarios = 250
for batch in range(24,batches):
    run_Model(instance, rl, num_Scenarios, MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], batch)