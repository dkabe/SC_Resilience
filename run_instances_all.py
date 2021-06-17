from stochastic_resilience import *
import time
instances = 5
MPs = [2, 3, 4, 6, 6]
DCs = [3, 4, 6, 8, 4]
MZs = [1, 2, 3, 5, 29]
numScenarios = [32, 128, 200, 200, 200]
epsilons = [200, 400, 600, 800, 350000]
rl = [0.5 , 0.55, 0.6 , 0.65, 0.7]
rl = [0.5 , 0.55, 0.6 , 0.65, 0.7, 0.75, 0.8 , 0.85, 0.9 , 0.95]

for instance in range(4,5):
    for r_level in rl:
            start_time = time.time()
            run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 1, 'f2': 1})
            end_time = time.time()
            print('CPU: ', end_time - start_time)
            print('\n')