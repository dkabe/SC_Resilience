from epsilon_constraint import *
import time
instances = 6
MPs = [2, 3, 4, 6, 6, 6]
DCs = [3, 4, 6, 8, 4, 4]
MZs = [1, 2, 3, 5, 29, 29]
numScenarios = [32, 128, 200, 200, 128, 200]
epsilons = [200, 400, 600, 800, 1500000, 700000]
rl = 0.75
instance = 4
resolution = 20

for res in range(resolution):
    run_Model(instance, rl, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], {'f1': 0, 'f2': 1}, 1, 0, res)
    print("Done")
