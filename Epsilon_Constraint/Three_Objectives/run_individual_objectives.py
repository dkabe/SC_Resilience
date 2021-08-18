from individual_objectives import *
import time
import multiprocessing as mp
instances = 2
MPs = [6, 6]
DCs = [4, 4]
MZs = [29, 29]
numScenarios = [128, 300]
epsilons = [1500000, 700000]
rl = [0.5, 0.75, 0.95]
dicts = [{'f1': 1, 'f2': 0, 'f3': 0}, {'f1': 0, 'f2': 1, 'f3': 0}, {'f1': 0, 'f2': 0, 'f3': 1}]

#for instance in range(1):
 #   for r_level in rl:
  #      for dct in dicts:
   #         start_time = time.time()
    #        run_Model(instance, r_level, numScenarios[instance], MPs[instance], DCs[instance], MZs[instance], Products[instance], Outsourced[instance], epsilons[instance], dct)
     #       end_time = time.time()
      #      print('CPU: ', end_time - start_time)
       #     print('\n')

with mp.Pool(3) as pool:
    pool.map(run_Model, dicts, chunksize=1)