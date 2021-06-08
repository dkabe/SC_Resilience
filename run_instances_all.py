from stochastic_resilience_v3 import *
import time
instances = 4
MPs = [2, 3, 4, 6]
DCs = [3, 4, 6, 8]
MZs = [1, 2, 3, 5]
numScenarios = [32, 128, 200, 200]
epsilons = [200, 400, 600, 800]
rl = [0.9, 0.8, 0.7, 0.5]

for instance in range(instances):
    for r_level in rl:
        start_time = time.time()
        run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products, Outsourced, epsilons[instance], {'f1': 0.7, 'f2': 0.3})
        end_time = time.time()
        print('CPU: ', end_time - start_time)
        print('\n')
