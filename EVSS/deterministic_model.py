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
numScenarios = [192, 192]

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

for instance in range(instances):
    text_file = open(path + 'Instance_' + str(instance + 1) + '/scen_' + str(instance + 1) + '.txt', "r")
    ls = text_file.read().split('\n')[:-1]
    Scen = list(map(lambda x: ast.literal_eval(x), ls))
    Scenarios.append(Scen)

# Initialize model variables

x_i = {} # opening manufacturing plant
x_j = {} # opening DC
U_km = {} # quantity lost sales
V1_lm = {} # quantity products purchased from outsourcing below epsilon threshold
V2_lm = {} # quantity products purchased from outsourcing in excess of epsilon threshold
Q_im = {} # quantity of product m produced at plant i
Y_ijm = {} # shipping i -> j
Z_jkm = {} # shipping j -> k
T_lkm = {} # shipping l -> k
w_s = {} # penalty for not meeting demand above specified rate

# variable values
v_val_x_i = {}
v_val_x_j = {}
v_val_U_km = {}
v_val_V1_lm = {}
v_val_V2_lm = {}
v_val_Q_im = {}
v_val_Y_ijm = {}
v_val_Z_jkm = {}
v_val_T_lkm = {}
v_val_w = {}

# Dictionaries for analysis
Cost_dict = {}
Summary_dict = {}

# Dictionary to weigh different objectives
objWeights = {}

# Dictionary to save values of each objectives
dic_grbOut = {}

grbModel_det = Model('deterministic')

def SetGurobiModel_det(instance, rl, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1):

    for i in range(Manufacturing_plants):
        x_i[i] = grbModel_det.addVar(vtype = GRB.BINARY)

    for j in range(Distribution):
        x_j[j] = grbModel_det.addVar(vtype = GRB.BINARY)

    for k in range(Market):
        for m in range(Products):
            U_km[k,m] = grbModel_det.addVar(vtype = GRB.CONTINUOUS)

    for m in range(Products):
        for l in range(Outsourced):
            V1_lm[m,l] = grbModel_det.addVar(vtype = GRB.CONTINUOUS)

    for m in range(Products):
        for l in range(Outsourced):
            V2_lm[m,l] = grbModel_det.addVar(vtype = GRB.CONTINUOUS)

    for m in range(Products):
        for i in range(Manufacturing_plants):
            Q_im[m,i] = grbModel_det.addVar(vtype = GRB.CONTINUOUS)

    for m in range(Products):
        for i in range(Manufacturing_plants):
            for j in range(Distribution):
                Y_ijm[m,i,j] = grbModel_det.addVar(vtype = GRB.CONTINUOUS)

    for m in range(Products):
        for j in range(Distribution):
            for k in range(Market):
                Z_jkm[m,j,k] = grbModel_det.addVar(vtype = GRB.CONTINUOUS)
    

    for m in range(Products):
        for l in range(Outsourced):
            for k in range(Market):
                    T_lkm[m,l,k] = grbModel_det.addVar(vtype = GRB.CONTINUOUS)

    for k in range(Market):
        for m in range(Products):
            w_s[k,m] = grbModel_det.addVar(vtype = GRB.CONTINUOUS)

    SetGrb_Obj_det(instance, rl, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1)
    ModelCons_det(instance, rl, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1)

def SolveModel_det():
    grbModel_det.params.OutputFlag = 0
    grbModel_det.optimize()
    global v_val_x_i
    global v_val_x_j
    v_val_x_i = grbModel_det.getAttr('x', x_i)
    v_val_x_j = grbModel_det.getAttr('x', x_j)
    Summary_dict['Obj'] = grbModel_det.objval
    
    return

# Objective

def SetGrb_Obj_det(instance, rl, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1):

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

    ship_1 = 0
    ship_2 = 0
    ship_3 = 0
    ship_4 = 0

    for i in range(Manufacturing_plants):
        for j in range(Distribution):
            for m in range(Products):
                ship_1 += Transportation_i_j[instance][m][i][j]*Y_ijm[m,i,j]

    for j in range(Distribution):
        for k in range(Market):
            for m in range(Products):
                ship_2 += Transportation_j_k[instance][m][j][k]*Z_jkm[m,j,k]
    

    for l in range(Outsourced):
        for k in range(Market):
            for m in range(Products):
                ship_4 += T_O_MZ[instance][m][l][k]*T_lkm[m,l,k]

    total_shipment += ship_1 + ship_2 + ship_3 + ship_4

    # Production
    pr_cost = 0
    for i in range(Manufacturing_plants):
        for m in range(Products):
            pr_cost += Manufacturing_costs[instance][i][m]*Q_im[m,i]

    total_pr_cost += pr_cost

    # Buying from outsource cost
    b_cost = 0
    for l in range(Outsourced):
        for m in range(Products):
            b_cost += Supplier_cost[instance][0][m][l]*V1_lm[m,l] + Supplier_cost[instance][1][m][l]*V2_lm[m,l]

    total_b_cost += b_cost

    #Lost Sales
    l_cost = 0
    for k in range(Market):
        for m in range(Products):
            l_cost += lost_sales[instance][k][m]*U_km[k,m]

    total_l_cost += l_cost

    # Penalties 
    rl_penalty = 0
    for k in range(Market):
        for m in range(Products):
            rl_penalty += lost_sales[instance][k][m]*w_s[k,m]*demand[instance][s1][m][k]

    grb_expr += OC_1 + OC_2 + (total_shipment + total_pr_cost + total_b_cost + total_l_cost + rl_penalty)

    grbModel_det.setObjective(grb_expr, GRB.MINIMIZE)

    return

def ModelCons_det(instance, rl, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1):

    # Network Flow

    grbModel_det.addConstrs(Q_im[m,i] >= quicksum(Y_ijm[m,i,j] for j in range(Distribution))
                         for i in range(Manufacturing_plants) for m in range(Products))

    grbModel_det.addConstrs(quicksum(Y_ijm[m,i,j] for i in range(Manufacturing_plants)) >= quicksum(Z_jkm[m,j,k] for k in range(Market))
                        for j in range(Distribution) for m in range(Products))

    grbModel_det.addConstrs((quicksum(Z_jkm[m,j,k] for j in range(Distribution)) +
                         quicksum(T_lkm[m,l,k] for l in range(Outsourced)) + U_km[k,m]) >= demand[instance][s1][m][k]
                         for k in range(Market) for m in range(Products))


    # Purchasing Constraints (everything purchased from outsourced facilities must be shipped)
    grbModel_det.addConstrs(V1_lm[m,l] + V2_lm[m,l] >= quicksum(T_lkm[m,l,k] for k in range(Market))
                        for m in range(Products) for l in range(Outsourced))

    # Capacity Constraints
    grbModel_det.addConstrs(quicksum(volume[instance][m]*Q_im[m,i] for m in range(Products)) <= Scenarios[instance][s1][0][i]*Capacities_i[instance][i]*x_i[i]
                         for i in range(Manufacturing_plants))

    grbModel_det.addConstrs(quicksum(volume[instance][m]*Y_ijm[m,i,j] for i in range(Manufacturing_plants) for m in range(Products)) <=
                        Scenarios[instance][s1][1][j]*Capacities_j[instance][j]*x_j[j]
                        for j in range(Distribution))

    grbModel_det.addConstrs((V1_lm[m,l] + V2_lm[m,l] <= (Capacities_l[instance][m][l]))
                        for l in range(Outsourced) for m in range(Products))


    # Indicator variable constraints for step function 
    grbModel_det.addConstrs(V1_lm[m,l] <= epsilon
                               for m in range(Products) for l in range(Outsourced))

    
    # Resilience Metric 
    grbModel_det.addConstrs(w_s[k,m] >= rl - (1 - U_km[k,m]/demand[instance][s1][m][k]) for k in range(Market) for m in range(Products))

    return

def save_FirstStageDecisions(instance, s1, rl):
    values = ["v_val_x_i", "v_val_x_j"]
    v_values = [v_val_x_i, v_val_x_j]
    ff = open("/home/dkabe/Model_brainstorming/EVSS/First_Stage_Decisions/" + "Instance_" + str(instance + 1) +  "/first_stage_" + str(s1) + "_" + str(rl) + ".txt", "w+")
    for i in range(len(v_values)):
        if i != len(v_values) - 1:
            ff.write(str(v_values[i]) + '\n')
        else:
            ff.write(str(v_values[i]))
    ff.close()
    return

def save_objectives(instance, rl):
    ff = open("/home/dkabe/Model_brainstorming/EVSS/EVPI_objectives/" + "Instance_" + str(instance + 1) + "/objectives_" + str(rl) + ".txt", "a")
    ff.write(str(Summary_dict['Obj']))
    ff.write('\n')
    ff.close()
    return

def run_Model_det(instance, rl, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1):

    SetGurobiModel_det(instance, rl, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, s1)
    SolveModel_det()
    save_FirstStageDecisions(instance, s1, rl) 
    save_objectives(instance, rl)

