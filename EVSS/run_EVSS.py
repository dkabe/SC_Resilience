from stochastic_model_EVSS import *
import multiprocessing as mp
import time
instances = 2
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
numScenarios = [128, 200]
epsilons = [1500000, 700000]
rl = [0.5]


with mp.Pool(32) as pool:
   pool.map(run_Model, [s for s in range(128)], chunksize=4)