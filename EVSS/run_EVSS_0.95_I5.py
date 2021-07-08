from stochastic_model_EVSS import *
import time
instances = 6
MPs = [2, 3, 4, 6, 6, 6]
DCs = [3, 4, 6, 8, 4, 4]
MZs = [1, 2, 3, 5, 29, 29]
numScenarios = [32, 128, 200, 200, 128, 200]
epsilons = [200, 400, 600, 800, 1500000, 700000]
rl = [0.95]

#for instance in range(4,5):    
#    for r_level in rl:
#        for scen in range(27):
#            run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], scen)

#from stochastic_model_EVSS import *
#for instance in range(4,5):    
#    for r_level in rl:
#        for scen in range(27, 54):
#            run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], scen)

for instance in range(4,5):    
    for r_level in rl:
        for scen in range(54, 81):
            run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], scen)

from stochastic_model_EVSS import *
for instance in range(4,5):    
    for r_level in rl:
        for scen in range(81, 108):
            run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], scen)

from stochastic_model_EVSS import *
for instance in range(4,5):    
    for r_level in rl:
        for scen in range(108, numScenarios[instance]):
            run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], scen)