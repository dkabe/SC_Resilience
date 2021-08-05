from deterministic_model import *
import time
instances = 2
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
numScenarios = [128, 200]
epsilons = [1500000, 700000]
rl = [0.5, 0.75, 0.95]

for instance in range(4,6):    
    for r_level in rl:
        for scen in range(numScenarios[instance]):
            start_time = time.time()
            run_Model_det(instance, r_level, MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], scen)
            end_time = time.time()
            print('CPU: ', end_time - start_time)
            print('\n')