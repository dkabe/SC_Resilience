from stochastic_objectives import *
import time
instances = 6
MPs = [2, 3, 4, 6, 6, 6]
DCs = [3, 4, 6, 8, 4, 4]
MZs = [1, 2, 3, 5, 29, 29]
numScenarios = [32, 128, 200, 200, 128, 200]
epsilons = [200, 400, 600, 800, 1500000, 700000]
rl = [0.5, 0.75]
dicts = [{'f1': 1, 'f2': 0}, {'f1': 0, 'f2': 1}]

for instance in range(4,5):
    for r_level in rl:
        for dct in dicts:
            start_time = time.time()
            run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], dct)
            end_time = time.time()
            print('CPU: ', end_time - start_time)
            print('\n')