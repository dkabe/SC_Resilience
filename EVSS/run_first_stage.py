from deterministic_model import *
import time
instances = 6
MPs = [2, 3, 4, 6, 6, 6]
DCs = [3, 4, 6, 8, 4, 4]
MZs = [1, 2, 3, 5, 29, 29]
numScenarios = [32, 128, 200, 200, 128, 200]
epsilons = [200, 400, 600, 800, 1500000, 700000]
rl = [0.95]

for instance in range(4,6):    
    for r_level in rl:
        for scen in range(numScenarios[instance]):
            start_time = time.time()
            run_Model_det(instance, r_level, MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], scen)
            end_time = time.time()
            print('CPU: ', end_time - start_time)
            print('\n')