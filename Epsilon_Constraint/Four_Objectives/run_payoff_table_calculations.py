from payoff_table_calculations import *
import time
instances = 6
MPs = [2, 3, 4, 6, 6, 6]
DCs = [3, 4, 6, 8, 4, 4]
MZs = [1, 2, 3, 5, 29, 29]
numScenarios = [32, 128, 200, 200, 128, 200]
epsilons = [200, 400, 600, 800, 1500000, 700000]
rl = 0.95
instance = 4

#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 1, 'f3': 0, 'f4': 0}, 1, 0, 0, 0, 0, 0, 0, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 0, 'f3': 1, 'f4': 0}, 1, 0, 0, 0, 0, 1, 0, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 0, 'f3': 0, 'f4': 1}, 1, 0, 0, 0, 0, 1, 1, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 1, 'f2': 0, 'f3': 0, 'f4': 0}, 0, 1, 0, 0, 0, 0, 0, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 0, 'f3': 1, 'f4': 0}, 0, 1, 0, 0, 1, 0, 0, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 0, 'f3': 0, 'f4': 1}, 0, 1, 0, 0, 1, 0, 1, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 1, 'f2': 0, 'f3': 0, 'f4': 0}, 0, 0, 1, 0, 0, 0, 0, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 1, 'f3': 0, 'f4': 0}, 0, 0, 1, 0, 1, 0, 0, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

# infeasible ?? 
#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 0, 'f3': 0, 'f4': 1}, 0, 0, 1, 0, 1, 1, 0, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 1, 'f2': 0, 'f3': 0, 'f4': 0}, 0, 0, 0, 1, 0, 0, 0, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

#start_time = time.time()
#run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 1, 'f3': 0, 'f4': 0}, 0, 0, 0, 1, 1, 0, 0, 0)
#end_time = time.time()
#print('CPU: ', end_time - start_time)
#print('\n')

start_time = time.time()
run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 0, 'f3': 1, 'f4': 0}, 0, 0, 0, 1, 1, 1, 0, 0)
end_time = time.time()
print('CPU: ', end_time - start_time)
print('\n')