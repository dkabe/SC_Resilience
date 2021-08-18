from stochastic_model_EVSS import *
import multiprocessing as mp
import time
instances = 2
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
numScenarios = [128, 300]
epsilons = [1500000, 700000]
rl = [0.5]


with mp.Pool(30) as pool:
   pool.map(run_Model, [s for s in range(240,300)], chunksize = 2)