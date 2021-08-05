from stochastic_model_EVSS import *
import time
instances = 2
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
numScenarios = [128, 200]
epsilons = [1500000, 700000]
rl = [0.95]

for instance in range(1,2):    
    for r_level in rl:
        for scen in range(196, numScenarios[instance]):
            run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], scen)
            grbModel.reset()

#from stochastic_model_EVSS import *
#for instance in range(4,5):    
 #   for r_level in rl:
  #      for scen in range(81, 108):
   #         run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], scen)

#for instance in range(4,5):    
 #   for r_level in rl:
  #      for scen in range(102, numScenarios[instance]):
   #         run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], scen)
    #        grbModel.reset()