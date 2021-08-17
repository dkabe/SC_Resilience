import numpy as np
import random
from random import randint
from gurobipy import *
import pandas as pd
from random import seed
import time
import ast

# Read input files
#path = "C:/Users/Devika Kabe/Documents/Model_brainstorming/Input_Data/"
path = "/home/dkabe/Model_brainstorming/Input_Data/Realistic/"
p_failure = 0.1
p_running = 1 - p_failure
instances = 2
Products  = [3,3]
Outsourced =[3,3]

levels = 2

Manufacturing_plants = [6, 6]
Distribution = [4, 4]
Market = [29, 29]
numScenarios = [128, 300]

# Read and append input files
f_i = [None]*instances
f_j = [None]*instances
volume = [None]*instances
Supplier_cost = [None]*instances 
Manufacturing_costs = [None]*instances
Transportation_i_j = [None]*instances
Transportation_j_k = [None]*instances
Capacities_i = [None]*instances
Capacities_j = [None]*instances
Capacities_l = [None]*instances
T_O_DC = [None]*instances
T_O_MZ = [None]*instances
lost_sales = [None]*instances
demand = [None]*instances

for instance in range(instances):
    # Cost of Opening
    f_i[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/OpenMP_' + str(instance + 1) + '.txt')
    f_j[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/OpenDC_' + str(instance + 1) + '.txt')

    # Unit cost of Manufacturing
    Manufacturing_costs[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/Manufacturing_' + str(instance + 1) + '.txt')

    # Transportation Costs
    Transportation_i_j[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/TransMPDC_' + str(instance + 1) + '.txt').reshape((Products[instance], Manufacturing_plants[instance], Distribution[instance]))
    Transportation_j_k[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/TransDCMZ_' + str(instance + 1) + '.txt').reshape((Products[instance], Distribution[instance], Market[instance]))

    # Plant Capacities
    Capacities_i[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/CapacitiesMP_' + str(instance + 1) + '.txt')
    Capacities_j[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/CapacitiesDC_' + str(instance + 1) + '.txt')
    Capacities_l[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/CapacitiesOutsource_' + str(instance + 1) + '.txt')

    # Cost of shipping from supplier
    T_O_DC[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/TransSupplierDC_' + str(instance + 1) + '.txt').reshape((Products[instance], Outsourced[instance], Distribution[instance]))
    T_O_MZ[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/TransSupplierMZ_' + str(instance + 1) + '.txt').reshape((Products[instance], Outsourced[instance], Market[instance]))

    # Unit cost of lost sales
    lost_sales[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/LostSales_' + str(instance + 1) + '.txt').reshape((Market[instance], Products[instance]))

    # demand
    demand[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/Demand_' + str(instance + 1) + '.txt').reshape((numScenarios[instance], Products[instance], Market[instance]))

    # volume
    volume[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/Volume_' + str(instance + 1) + '.txt')

    # Supplier cost
    Supplier_cost[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/SupplierCost_' + str(instance + 1) + '.txt').reshape((levels, Products[instance], Outsourced[instance]))

Scenarios = []
Probabilities = []

for instance in range(instances):
    text_file = open(path + 'Instance_' + str(instance + 1) + '/scen_' + str(instance + 1) + '.txt', "r")
    ls = text_file.read().split('\n')[:-1]
    Scen = list(map(lambda x: ast.literal_eval(x), ls))
    p_scen = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/p_scen_' + str(instance + 1) + '.txt')
    Scenarios.append(Scen)
    Probabilities.append(p_scen)



# Initialize model variables

grbModel = Model()

x_i = {} # opening manufacturing plant
x_j = {} # opening DC
U_km = {} # quantity lost sales
V1_lm = {} # quantity products purchased from outsourcing below epsilon threshold
V2_lm = {} # quantity products purchased from outsourcing in excess of epsilon threshold
Q_im = {} # quantity of product m produced at plant i
Y_ijm = {} # shipping i -> j
Z_jkm = {} # shipping j -> k
T_ljm = {} # shipping l -> j
T_lkm = {} # shipping l -> k
w_s = {} # penalty for not meeting demand above specified rate


# Dictionaries for analysis
Cost_dict = {}
Summary_dict = {}

# Dictionary to weigh different objectives
objWeights = {}

# Dictionary to save values of each objectives
dic_grbOut = {}


def InitializeModelParams(instance, s1, rl):
    global x_i
    global x_j
    path = '/home/dkabe/Model_brainstorming/EVSS/First_Stage_Decisions/'
    f = open(path + "Instance_" + str(instance + 1) + "/first_stage_" + str(s1) + "_" + str(rl) + ".txt", "r")
    text = f.read()
    f.close()
    solutions_str = text.split('\n')
    x_i = ast.literal_eval(solutions_str[0])
    x_j = ast.literal_eval(solutions_str[1])
    return

def SetGurobiModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1):

    global U_km 
    global V1_lm 
    global V2_lm 
    global Q_im 
    global Y_ijm 
    global Z_jkm 
    global T_ljm 
    global T_lkm 
    global w_s 

 
    U_km = grbModel.addVars(range(num_Scenarios), range(Market), range(Products), vtype = GRB.CONTINUOUS)    
    V1_lm = grbModel.addVars(range(num_Scenarios), range(Products), range(Outsourced), vtype = GRB.CONTINUOUS)
    V2_lm = grbModel.addVars(range(num_Scenarios), range(Products), range(Outsourced), vtype = GRB.CONTINUOUS)
    Q_im = grbModel.addVars(range(num_Scenarios), range(Products), range(Manufacturing_plants), vtype = GRB.CONTINUOUS)
    Y_ijm = grbModel.addVars(range(num_Scenarios), range(Products), range(Manufacturing_plants), range(Distribution), vtype = GRB.CONTINUOUS)
    Z_jkm = grbModel.addVars(range(num_Scenarios), range(Products), range(Distribution), range(Market), vtype = GRB.CONTINUOUS)
    T_ljm = grbModel.addVars(range(num_Scenarios), range(Products), range(Outsourced), range(Distribution), vtype = GRB.CONTINUOUS)
    T_lkm = grbModel.addVars(range(num_Scenarios), range(Products), range(Outsourced), range(Market), vtype = GRB.CONTINUOUS)
    w_s = grbModel.addVars(range(num_Scenarios), range(Market), range(Products), vtype = GRB.CONTINUOUS)


    SetGrb_Obj(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, s1)
    ModelCons(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1)

def SolveModel(s1, instance):
    grbModel.params.OutputFlag = 0
    grbModel.optimize()

    Summary_dict['obj'] = Probabilities[instance][s1]*grbModel.objval

    return

# Objective

def SetGrb_Obj(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, s1):

    grb_expr = LinExpr()

    # Cost of opening
    OC_1 = 0
    OC_2 = 0
    for i in range(Manufacturing_plants):
        OC_1 += f_i[instance][i]*x_i[i]
    for j in range(Distribution):
        OC_2 += f_j[instance][j]*x_j[j]

    total_shipment = 0
    total_pr_cost = 0
    total_b_cost = 0
    total_l_cost = 0

    # Shipment

    for s in range(num_Scenarios):
        ship_1 = 0
        ship_2 = 0
        ship_3 = 0
        ship_4 = 0
        for i in range(Manufacturing_plants):
            for j in range(Distribution):
                for m in range(Products):
                    ship_1 += Transportation_i_j[instance][m][i][j]*Y_ijm[s,m,i,j]

        for j in range(Distribution):
            for k in range(Market):
                for m in range(Products):
                    ship_2 += Transportation_j_k[instance][m][j][k]*Z_jkm[s,m,j,k]

        for l in range(Outsourced):
            for j in range(Distribution):
                for m in range(Products):
                    ship_3 += T_O_DC[instance][m][l][j]*T_ljm[s,m,l,j]

        for l in range(Outsourced):
            for k in range(Market):
                for m in range(Products):
                    ship_4 += T_O_MZ[instance][m][l][k]*T_lkm[s,m,l,k]

        total_shipment += Probabilities[instance][s]*(ship_1 + ship_2 + ship_3 + ship_4)

        # Production
        pr_cost = 0
        for i in range(Manufacturing_plants):
            for m in range(Products):
                pr_cost += Manufacturing_costs[instance][i][m]*Q_im[s,m,i]

        total_pr_cost += Probabilities[instance][s]*pr_cost

        # Buying from outsource cost
        b_cost = 0
        for l in range(Outsourced):
            for m in range(Products):
                b_cost += Supplier_cost[instance][0][m][l]*V1_lm[s,m,l] + Supplier_cost[instance][1][m][l]*V2_lm[s,m,l]

        total_b_cost += Probabilities[instance][s]*b_cost

        #Lost Sales
        l_cost = 0
        for k in range(Market):
            for m in range(Products):
                l_cost += lost_sales[instance][k][m]*U_km[s,k,m]

        total_l_cost += Probabilities[instance][s]*l_cost
        
    rl_penalty = 0
    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                rl_penalty += Probabilities[instance][s]*lost_sales[instance][k][m]*w_s[s,k,m]*demand[instance][s][m][k]

    grb_expr += (OC_1 + OC_2 + (total_shipment + total_pr_cost + total_b_cost + total_l_cost + rl_penalty))

    grbModel.setObjective(grb_expr, GRB.MINIMIZE)

    return

# Model Constraints

def ModelCons(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1):

    # Network Flow

    grbModel.addConstrs(Q_im[s,m,i] >= quicksum(Y_ijm[s,m,i,j] for j in range(Distribution))
                         for s in range(num_Scenarios) for i in range(Manufacturing_plants) for m in range(Products))

    grbModel.addConstrs((quicksum(Y_ijm[s,m,i,j] for i in range(Manufacturing_plants)) +
                         quicksum(T_ljm[s,m,l,j] for l in range(Outsourced))) >= quicksum(Z_jkm[s,m,j,k] for k in range(Market))
                        for s in range(num_Scenarios) for j in range(Distribution) for m in range(Products))

    grbModel.addConstrs((quicksum(Z_jkm[s,m,j,k] for j in range(Distribution)) +
                         quicksum(T_lkm[s,m,l,k] for l in range(Outsourced)) + U_km[s,k,m]) >= demand[instance][s][m][k]
                         for s in range(num_Scenarios) for k in range(Market) for m in range(Products))


    # Purchasing Constraints (everything purchased from outsourced facilities must be shipped)
    grbModel.addConstrs(V1_lm[s,m,l] + V2_lm[s,m,l] >= quicksum(T_ljm[s,m,l,j] for j in range(Distribution)) +
                        quicksum(T_lkm[s,m,l,k] for k in range(Market)) for s in range(num_Scenarios)
                        for m in range(Products) for l in range(Outsourced))


    # Capacity Constraints
    grbModel.addConstrs(quicksum(volume[instance][m]*Q_im[s,m,i] for m in range(Products)) <= Scenarios[instance][s][0][i]*Capacities_i[instance][i]*x_i[i]
                        for s in range(num_Scenarios) for i in range(Manufacturing_plants))

    grbModel.addConstrs(quicksum(volume[instance][m]*Y_ijm[s,m,i,j] for i in range(Manufacturing_plants) for m in range(Products)) +
                        quicksum(volume[instance][m]*T_ljm[s,m,l,j] for l in range(Outsourced) for m in range(Products)) <=
                        Scenarios[instance][s][1][j]*Capacities_j[instance][j]*x_j[j] for s in range(num_Scenarios) for s in range(num_Scenarios)
                        for j in range(Distribution))

    grbModel.addConstrs(V1_lm[s,m,l] + V2_lm[s,m,l] <= Capacities_l[instance][m][l] for s in range(num_Scenarios)
                        for l in range(Outsourced) for m in range(Products))


    # constraints for step function epsilon
    grbModel.addConstrs(V1_lm[s,m,l] <= epsilon for s in range(num_Scenarios)
                        for l in range(Outsourced) for m in range(Products))

    # Resilience Metric (w = % of rl being missed)

    grbModel.addConstrs(w_s[s,k,m] >= rl - (1 - U_km[s,k,m]/demand[instance][s][m][k]) for s in range(num_Scenarios) for k in range(Market) for m in range(Products))

    

    return

def save_results(instance, rl):
    f = open("/home/dkabe/Model_brainstorming/EVSS/V_Det/" + "Instance_" + str(instance + 1) + "/v_det_" + str(rl) + ".txt", "a")
    f.write(str(Summary_dict['obj'])) 
    f.write('\n')
    f.close()

    return

def run_Model(s1, instance=1, rl=0.95, num_Scenarios=300, Manufacturing_plants=6, Distribution=4, Market=29, Products=3, Outsourced=3, epsilon=700000):
    InitializeModelParams(instance, s1, rl)
    SetGurobiModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1)
    SolveModel(s1, instance)
    save_results(instance, rl)