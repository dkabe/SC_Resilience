from stochastic_resilience_v2 import *
import time
instances = 2
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
numScenarios = [128, 200]
epsilons = [1500000, 700000]
rl = [0.5 , 0.55, 0.6 , 0.65, 0.7, 0.75, 0.8 , 0.85, 0.9 , 0.95]
for instance in range(2):
    for r_level in rl:
        start_time = time.time()
        run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 1, 'f2': 1}, 0)
        end_time = time.time()
        print('CPU: ', end_time - start_time)
        print('\n')
