from epsilon_constraint import *
import time
instances = 2
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
numScenarios = [128, 200]
epsilons = [1500000, 700000]
rl = 0.75
instance = 0
resolution = 20

for res in range(resolution):
    run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 1}, 1, 0, res)
    print("Done")
