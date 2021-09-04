from payoff_table_calculations import *
import time
instances = 2
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
numScenarios = [192, 192]
epsilons = [1500000, 700000]
rl = 0.5
instance = 1

start_time = time.time()
run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 1}, 1, 0)
end_time = time.time()
print('CPU: ', end_time - start_time)
#print('\n')

start_time = time.time()
run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 1, 'f2': 0}, 0, 1)
end_time = time.time()
print('CPU: ', end_time - start_time)
#print('\n')