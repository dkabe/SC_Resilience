from epsilon_constraint import *
import time
import multiprocessing as mp
instances = 2
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
numScenarios = [128, 300]
epsilons = [1500000, 700000]
rl = 0.5
instance = 0
resolution = range(10)

#for res in range(resolution):
 #   for res2 in range(resolution):

with mp.Pool(40) as pool:
    pool.starmap(run_Model, [(e1, e2) for e1 in resolution for e2 in resolution], chunksize=3)
