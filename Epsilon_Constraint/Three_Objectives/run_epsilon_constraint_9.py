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
instance = 1
resolution = range(20)
e_list = [(e1, e2) for e1 in resolution for e2 in resolution]



#with mp.Pool(40) as pool:
 #   pool.starmap(run_Model, e_list[:40])
  #  pool.close()

#with mp.Pool(40) as pool:
 #   pool.starmap(run_Model, e_list[40:80], chunksize=1)
  #  pool.close()

#with mp.Pool(40) as pool:
 #   pool.starmap(run_Model, e_list[80:120], chunksize=1)
  #  pool.close()   

#with mp.Pool(40) as pool:
 #   pool.starmap(run_Model, e_list[120:160], chunksize=1)
  #  pool.close() 
  
#with mp.Pool(40) as pool:
 #   pool.starmap(run_Model, e_list[160:200], chunksize=1)
  #  pool.close()   
 
#with mp.Pool(40) as pool:
 #   pool.starmap(run_Model, e_list[200:240], chunksize=1)
  #  pool.close() 

#with mp.Pool(40) as pool:
 #   pool.starmap(run_Model, e_list[240:280], chunksize=1)
  #  pool.close() 

#with mp.Pool(40) as pool:
   # pool.starmap(run_Model, e_list[280:320], chunksize=1)
  #  pool.close() 

with mp.Pool(40) as pool:
    pool.starmap(run_Model, e_list[320:360], chunksize=1)
    pool.close()   

#with mp.Pool(40) as pool:
 #   pool.starmap(run_Model, e_list[360:400], chunksize=1)
  #  pool.close()   