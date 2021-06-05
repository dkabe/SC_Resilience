from stochastic_resilience_v3 import *
import time
instances = 1
MPs = [2, 3, 4, 6]
DCs = [3, 4, 6, 8]
MZs = [1, 2, 3, 5]
numScenarios = [32, 128, 200, 200]
epsilons = [200, 400, 600, 800]

for instance in range(2,3):
    start_time = time.time()
    run_Model(instance, 0.9, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products, Outsourced, {'f1': 0.7, 'f2': 0.3})
    end_time = time.time()
    print('CPU: ', end_time - start_time)
    print('\n')
