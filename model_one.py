from gurobipy import *
import numpy as np
import sys

# Model Sets
Manufacturing_plants = 2
Distribution = 3
Market = 4
Products = 2
Outsourced = 2

# Model Parameters

# Disruption binary vector
a_i = np.ones(Manufacturing_plants)
b_i = np.ones(Distribution)

# Demand
Demand = 
