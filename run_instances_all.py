import math
import numpy as np
import random
from random import randint
from gurobipy import *
import pandas as pd
from random import seed
import matplotlib.pyplot as plt

# Read input files
path = "C:/Users/Devika Kabe/Documents/Model_brainstorming/Input_Data/Instance_4/"

# Cost of Opening
f_i = np.loadtxt(path + 'OpenMP_4.txt')
f_j = np.loadtxt(path + 'OpenDC_4.txt')

# Unit cost of Manufacturing
Manufacturing_costs = np.loadtxt(path + 'Manufacturing_4.txt')

# Transportation Costs
Transportation_i_j = np.loadtxt(path + 'TransMPDC_4.txt').reshape((Products, Manufacturing_plants, Distribution))
Transportation_j_k = np.loadtxt(path + 'TransDCMZ_4.txt').reshape((Products, Distribution, Market))

# Plant Capacities
Capacities_i = np.loadtxt(path + 'CapacitiesMP_4.txt')
Capacities_j = np.loadtxt(path + 'CapacitiesDC_4.txt')
Capacities_l = np.loadtxt(path + 'CapacitiesOutsource_4.txt')

# Cost of purchasing from supplier
Supplier_cost = np.loadtxt(path + 'SupplierCost_4.txt').reshape((levels, Products, Outsourced))

# Cost of shipping from supplier
T_O_DC = np.loadtxt(path + 'TransSupplierDC_4.txt').reshape((Products, Outsourced, Distribution))
T_O_MZ = np.loadtxt(path + 'TransSupplierMZ_4.txt').reshape((Products, Outsourced, Market))

# volume of product
volume = np.loadtxt(path + 'Volume_4.txt')

# Unit cost of lost sales
lost_sales = np.loadtxt(path + 'LostSales_4.txt')

# demand
demand = np.loadtxt(path + 'Demand_4.txt').reshape(num_Scenarios, Products, Market)
s_Demand = sum(demand)
