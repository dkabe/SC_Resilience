from gurobipy import *
import math
import time
import numpy as np
import random
from random import randint, seed
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import ast 

# Read input files
#path = "C:/Users/Devika Kabe/Documents/Model_brainstorming/Input_Data/"
path = "/home/dkabe/Model_brainstorming/Input_Data/Realistic/"

instances = 2

levels = 2


# Initialize parameters (modified in function below)

# Cost of Opening
f_i = []
f_j = []

# volume of products
volume = []

# supplier cost
Supplier_cost = [] 

# Manufacturing products cost
Manufacturing_costs = []

# Transportation costs 
Transportation_i_j = []
Transportation_j_k = []
T_O_DC = []
T_O_MZ = []

# Capacities of facilities (square footage)
Capacities_i = []
Capacities_j = []

# Availability of products in outsourced facilities
Capacities_l = []

# lost sales
lost_sales = []

# scenario demand 
demand = []

# Disruption scenarios
Scenarios = []

# Probability of scenarios 
Probabilities = []

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

# variable values
v_val_x_i = {}
v_val_x_j = {}
v_val_U_km = {}
v_val_V1_lm = {}
v_val_V2_lm = {}
v_val_Q_im = {}
v_val_Y_ijm = {}
v_val_Z_jkm = {}
v_val_T_ljm = {}
v_val_T_lkm = {}
v_val_w = {}

# Dictionaries for analysis
Cost_dict = {}
Summary_dict = {}

# Dictionary to weigh different objectives
objWeights = {}

# Dictionary to save values of each objectives
dic_grbOut = {}

def InitializeModelParams(instance, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced):

    global f_i 
    global f_j 
    global volume 
    global Supplier_cost  
    global Manufacturing_costs 
    global Transportation_i_j 
    global Transportation_j_k 
    global Capacities_i 
    global Capacities_j 
    global Capacities_l 
    global T_O_DC 
    global T_O_MZ 
    global lost_sales 
    global demand 
    global Scenarios
    global Probabilities

    f_i = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/OpenMP_' + str(instance + 1) + '.txt')
    f_j = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/OpenDC_' + str(instance + 1) + '.txt')
    Manufacturing_costs = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/Manufacturing_' + str(instance + 1) + '.txt')
    Transportation_i_j = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/TransMPDC_' + str(instance + 1) + '.txt').reshape((Products, Manufacturing_plants, Distribution))
    Transportation_j_k = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/TransDCMZ_' + str(instance + 1) + '.txt').reshape((Products, Distribution, Market))
    Capacities_i = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/CapacitiesMP_' + str(instance + 1) + '.txt')
    Capacities_j = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/CapacitiesDC_' + str(instance + 1) + '.txt')
    Capacities_l = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/CapacitiesOutsource_' + str(instance + 1) + '.txt')
    T_O_DC = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/TransSupplierDC_' + str(instance + 1) + '.txt').reshape((Products, Outsourced, Distribution))
    T_O_MZ = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/TransSupplierMZ_' + str(instance + 1) + '.txt').reshape((Products, Outsourced, Market))
    lost_sales = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/LostSales_' + str(instance + 1) + '.txt').reshape((Market, Products))
    demand = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/Demand_' + str(instance + 1) + '.txt').reshape((num_Scenarios, Products, Market))
    volume = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/Volume_' + str(instance + 1) + '.txt')
    Supplier_cost = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/SupplierCost_' + str(instance + 1) + '.txt').reshape((levels, Products, Outsourced))

    text_file = open(path + 'Instance_' + str(instance + 1) + '/scen_' + str(instance + 1) + '.txt', "r")
    ls = text_file.read().split('\n')[:-1]
    Scenarios = list(map(lambda x: ast.literal_eval(x), ls))
    Probabilities = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/p_scen_' + str(instance + 1) + '.txt')

    return

def SetGurobiModel(rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon):

    global x_i 
    global x_j 
    global U_km 
    global V1_lm 
    global V2_lm 
    global Q_im 
    global Y_ijm 
    global Z_jkm 
    global T_ljm 
    global T_lkm 
    global w_s 

    x_i = grbModel.addVars(range(Manufacturing_plants), vtype = GRB.BINARY)
    x_j = grbModel.addVars(range(Distribution), vtype = GRB.BINARY)              
    U_km = grbModel.addVars(range(num_Scenarios), range(Market), range(Products), vtype = GRB.CONTINUOUS)    
    V1_lm = grbModel.addVars(range(num_Scenarios), range(Products), range(Outsourced), vtype = GRB.CONTINUOUS)
    V2_lm = grbModel.addVars(range(num_Scenarios), range(Products), range(Outsourced), vtype = GRB.CONTINUOUS)
    Q_im = grbModel.addVars(range(num_Scenarios), range(Products), range(Manufacturing_plants), vtype = GRB.CONTINUOUS)
    Y_ijm = grbModel.addVars(range(num_Scenarios), range(Products), range(Manufacturing_plants), range(Distribution), vtype = GRB.CONTINUOUS)
    Z_jkm = grbModel.addVars(range(num_Scenarios), range(Products), range(Distribution), range(Market), vtype = GRB.CONTINUOUS)
    T_ljm = grbModel.addVars(range(num_Scenarios), range(Products), range(Outsourced), range(Distribution), vtype = GRB.CONTINUOUS)
    T_lkm = grbModel.addVars(range(num_Scenarios), range(Products), range(Outsourced), range(Market), vtype = GRB.CONTINUOUS)
    w_s = grbModel.addVars(range(num_Scenarios), range(Market), range(Products), vtype = GRB.CONTINUOUS)

    SetGrb_Obj(num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced)
    ModelCons(rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon)

def SolveModel(num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced):
    grbModel.params.OutputFlag = 0
    grbModel.params.timelimit = 900
    start_time = time.time()
    grbModel.optimize()
    #gap = grbModel.MIPGAP
    # get variable values

    global v_val_x_i 
    global v_val_x_j 
    global v_val_U_km 
    global v_val_V1_lm 
    global v_val_V2_lm 
    global v_val_Q_im 
    global v_val_Y_ijm 
    global v_val_Z_jkm 
    global v_val_T_ljm 
    global v_val_T_lkm 
    global v_val_w 

    v_val_x_i = grbModel.getAttr('x', x_i)
    v_val_x_j = grbModel.getAttr('x', x_j)
    v_val_U_km = grbModel.getAttr('x', U_km)
    v_val_V1_lm = grbModel.getAttr('x', V1_lm)
    v_val_V2_lm = grbModel.getAttr('x', V2_lm)
    v_val_Q_im = grbModel.getAttr('x', Q_im)
    v_val_Y_ijm = grbModel.getAttr('x', Y_ijm)
    v_val_Z_jkm = grbModel.getAttr('x', Z_jkm)
    v_val_T_ljm = grbModel.getAttr('x', T_ljm)
    v_val_T_lkm = grbModel.getAttr('x', T_lkm)
    v_val_w = grbModel.getAttr('x', w_s)

    Summary_dict['ObjVal'] = np.round(grbModel.objval,2)
    Summary_dict["OpenMPs"] = np.sum(v_val_x_i.values())
    Summary_dict["OpenDCs"] = np.sum(v_val_x_j.values())
    Cost_dict["Opening"] =  get_opening_costs(v_val_x_i, v_val_x_j, Manufacturing_plants, Distribution)
    Cost_dict["f4"] = np.round(get_rl_rate(v_val_w, num_Scenarios, Market, Products), 2)

    for s in range(num_Scenarios):
        Summary_dict["Purchasing_" + str(s)] = sum([v_val_V1_lm[(s,m,l)] +  v_val_V2_lm[(s,m,l)] for m in range(Products) for l in range(Outsourced)])
        Summary_dict["Production_" + str(s)] = sum([v_val_Q_im[(s,m,i)] for m in range(Products) for i in range(Manufacturing_plants)])
        Summary_dict["LostSales_" + str(s)] = sum([v_val_U_km[(s,k,m)] for m in range(Products) for k in range(Market)])
        Summary_dict["OutsourceToDC_" + str(s)] = sum([v_val_T_ljm[(s,m,l,j)] for m in range(Products) for l in range(Outsourced) for j in range(Distribution)])
        Summary_dict["OutsourceToMarket_" + str(s)] = sum([v_val_T_lkm[(s,m,l,k)] for m in range(Products) for l in range(Outsourced) for k in range(Market)])

    for s in range(num_Scenarios):
        Cost_dict["InHouseShipping_" + str(s)] = get_shipping_costs(s,v_val_Y_ijm, v_val_Z_jkm, v_val_T_ljm, v_val_T_lkm, Manufacturing_plants, Distribution, Products, Market, Outsourced)[0]
        Cost_dict["OutsourceShipping_" + str(s)] = get_shipping_costs(s,v_val_Y_ijm, v_val_Z_jkm, v_val_T_ljm, v_val_T_lkm, Manufacturing_plants, Distribution, Products, Market, Outsourced)[1]
        Cost_dict["Production_" + str(s)] = get_production_cost(s,v_val_Q_im, Manufacturing_plants, Products)
        Cost_dict["Purchasing_" + str(s)] = get_purchase_costs(s,v_val_V1_lm, v_val_V2_lm, Outsourced, Products)
        Cost_dict["LostSales_" + str(s)] = get_lost_cost(s,v_val_U_km, Market, Products)    

    f1_cost = 0
    f2_cost = 0
    f3_cost = 0
    for s in range(num_Scenarios):
        f1_cost += Probabilities[s]*(Cost_dict['Production_' + str(s)] + Cost_dict['InHouseShipping_' + str(s)])
        f2_cost += Probabilities[s]*(Cost_dict['Purchasing_' + str(s)] + Cost_dict['OutsourceShipping_' + str(s)])
        f3_cost +=  Probabilities[s]*Cost_dict['LostSales_' + str(s)]
    Cost_dict["f1"] = np.round(Cost_dict["Opening"] + f1_cost, 2) # in house (opening + production + shipping)
    Cost_dict["f2"] = np.round(f2_cost, 2) # Outsourcing # (purchasing + shipping)
    Cost_dict["f3"] = np.round(f3_cost, 2) # lost sales
    Summary_dict['Demand_met'] = np.sum([Probabilities[s]*(Summary_dict["Purchasing_" + str(s)] + Summary_dict["Production_" + str(s)])/np.sum(demand[s]) for s in range(num_Scenarios)])
    Summary_dict['Demand_outsourced'] = np.sum([Probabilities[s]*Summary_dict["Purchasing_" + str(s)]/np.sum(demand[s]) for s in range(num_Scenarios)])
    end_time = time.time()

    Summary_dict['CPU'] = end_time - start_time  
        
    return

# Objective

def SetGrb_Obj(num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced):
    grb_expr = LinExpr()

    # Cost of opening
    OC_1 = 0
    OC_2 = 0
    for i in range(Manufacturing_plants):
        OC_1 += f_i[i]*x_i[i]
    for j in range(Distribution):
        OC_2 += f_j[j]*x_j[j]

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
                    ship_1 += Transportation_i_j[m][i][j]*Y_ijm[s,m,i,j]

        for j in range(Distribution):
            for k in range(Market):
                for m in range(Products):
                    ship_2 += Transportation_j_k[m][j][k]*Z_jkm[s,m,j,k]

        for l in range(Outsourced):
            for j in range(Distribution):
                for m in range(Products):
                    ship_3 += T_O_DC[m][l][j]*T_ljm[s,m,l,j]

        for l in range(Outsourced):
            for k in range(Market):
                for m in range(Products):
                    ship_4 += T_O_MZ[m][l][k]*T_lkm[s,m,l,k]

        total_shipment += Probabilities[s]*(ship_1 + ship_2 + ship_3 + ship_4)

        # Production
        pr_cost = 0
        for i in range(Manufacturing_plants):
            for m in range(Products):
                pr_cost += Manufacturing_costs[i][m]*Q_im[s,m,i]

        total_pr_cost += Probabilities[s]*pr_cost

        # Buying from outsource cost
        b_cost = 0
        for l in range(Outsourced):
            for m in range(Products):
                b_cost += Supplier_cost[0][m][l]*V1_lm[s,m,l] + Supplier_cost[1][m][l]*V2_lm[s,m,l]

        total_b_cost += Probabilities[s]*b_cost

        #Lost Sales
        l_cost = 0
        for k in range(Market):
            for m in range(Products):
                l_cost += lost_sales[k][m]*U_km[s,k,m]

        total_l_cost += Probabilities[s]*l_cost

    # Percentage of demand met
    rl_penalty = 0
    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                rl_penalty += Probabilities[s]*lost_sales[k][m]*w_s[s,k,m]*demand[s][m][k]

    grb_expr += objWeights['f1']*(OC_1 + OC_2 + (total_shipment + total_pr_cost + total_b_cost + total_l_cost)) + objWeights['f2']*rl_penalty

    grbModel.setObjective(grb_expr, GRB.MINIMIZE)

    return

    # Model Constraints

def ModelCons(rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon):

    # Network Flow

    grbModel.addConstrs(Q_im[s,m,i] >= quicksum(Y_ijm[s,m,i,j] for j in range(Distribution))
                         for s in range(num_Scenarios) for i in range(Manufacturing_plants) for m in range(Products))

    grbModel.addConstrs((quicksum(Y_ijm[s,m,i,j] for i in range(Manufacturing_plants)) +
                         quicksum(T_ljm[s,m,l,j] for l in range(Outsourced))) >= quicksum(Z_jkm[s,m,j,k] for k in range(Market))
                        for s in range(num_Scenarios) for j in range(Distribution) for m in range(Products))

    grbModel.addConstrs(quicksum(Z_jkm[s,m,j,k] for j in range(Distribution)) +
                        quicksum(T_lkm[s,m,l,k] for l in range(Outsourced)) +
                        U_km[s,k,m] >= demand[s][m][k] for s in range(num_Scenarios) for m in range(Products)
                        for k in range(Market))


    # Purchasing Constraints (everything purchased from outsourced facilities must be shipped)
    grbModel.addConstrs(V1_lm[s,m,l] + V2_lm[s,m,l] >= quicksum(T_ljm[s,m,l,j] for j in range(Distribution)) +
                        quicksum(T_lkm[s,m,l,k] for k in range(Market)) for s in range(num_Scenarios)
                        for m in range(Products) for l in range(Outsourced))

    # Capacity Constraints
    grbModel.addConstrs(quicksum(volume[m]*Q_im[s,m,i] for m in range(Products)) <= Scenarios[s][0][i]*Capacities_i[i]*x_i[i]
                        for s in range(num_Scenarios) for i in range(Manufacturing_plants))

    grbModel.addConstrs(quicksum(volume[m]*Y_ijm[s,m,i,j] for i in range(Manufacturing_plants) for m in range(Products)) +
                        quicksum(volume[m]*T_ljm[s,m,l,j] for l in range(Outsourced) for m in range(Products)) <=
                        Scenarios[s][1][j]*Capacities_j[j]*x_j[j] for s in range(num_Scenarios) for s in range(num_Scenarios)
                        for j in range(Distribution))

    grbModel.addConstrs((V1_lm[s,m,l] + V2_lm[s,m,l] <= (Capacities_l[m][l])) for s in range(num_Scenarios)
                        for l in range(Outsourced) for m in range(Products))
    
    # Indicator variable constraints for step function 
    grbModel.addConstrs(V1_lm[s,m,l] <= epsilon for s in range(num_Scenarios) for m in range(Products) for l in range(Outsourced))


    # Resilience Metric (w = % of rl being missed)
    grbModel.addConstrs(w_s[s,k,m] >= rl - (1 - U_km[s,k,m]/demand[s][m][k]) for s in range(num_Scenarios) for k in range(Market) for m in range(Products))

    return

def get_opening_costs(x1, x2, Manufacturing_plants, Distribution):

    # Cost of opening
    OC_1 = 0
    OC_2 = 0
    for i in range(Manufacturing_plants):
        OC_1 += f_i[i]*x1[i]
    for j in range(Distribution):
        OC_2 += f_j[j]*x2[j]

    Opening = np.round(OC_1 + OC_2)

    return(Opening)

def get_shipping_costs(scen, Y, Z, T1, T2, Manufacturing_plants, Distribution, Products, Market, Outsourced):
    ship_1 = 0
    ship_2 = 0
    ship_3 = 0
    ship_4 = 0

    # Shipment
    for i in range(Manufacturing_plants):
        for j in range(Distribution):
            for m in range(Products):
                ship_1 += Transportation_i_j[m][i][j]*Y[scen, m,i,j]

    for j in range(Distribution):
        for k in range(Market):
            for m in range(Products):
                ship_2 += Transportation_j_k[m][j][k]*Z[scen,m,j,k]

    for l in range(Outsourced):
        for j in range(Distribution):
            for m in range(Products):
                ship_3 += T_O_DC[m][l][j]*T1[scen,m,l,j]

    for l in range(Outsourced):
        for k in range(Market):
            for m in range(Products):
                ship_4 += T_O_MZ[m][l][k]*T2[scen,m,l,k]

    in_house_shipping = np.round(ship_1 + ship_2)

    outsourced_shipping = np.round(ship_3 + ship_4)

    return(in_house_shipping, outsourced_shipping)

def get_production_cost(scen, Q, Manufacturing_plants, Products):

    # Production
    pr_cost = 0
    for i in range(Manufacturing_plants):
        for m in range(Products):
            pr_cost += Manufacturing_costs[i][m]*Q[scen,m,i]

    return(np.round(pr_cost))

def get_purchase_costs(scen, V1, V2, Outsourced, Products):

    # Buying from outsource cost
    b_cost = 0
    for l in range(Outsourced):
        for m in range(Products):
            b_cost += Supplier_cost[0][m][l]*V1[scen,m,l] + Supplier_cost[1][m][l]*V2[scen,m,l]

    return(np.round(b_cost))

def get_lost_cost(scen, U, Market, Products):

    #Lost Sales
    l_cost = 0
    for k in range(Market):
        for m in range(Products):
            l_cost += lost_sales[k][m]*U[scen,k,m]

    return(np.round(l_cost))

def get_outsourced_cost(scen, V1, V2, T1, T2, Distribution, Products, Outsourced, Market):
    # Buying from outsource cost
    b_cost = 0
    ship_to_distribution = 0
    ship_to_market = 0
    for l in range(Outsourced):
        for m in range(Products):
            b_cost += Supplier_cost[0][m][l]*V1[scen,m,l] + Supplier_cost[1][m][l]*V2[scen,m,l]

    # Shipping from outsourced cost
    for l in range(Outsourced):
        for j in range(Distribution):
            for m in range(Products):
                ship_to_distribution += T_O_DC[m][l][j]*T1[scen,m,l,j]

    for l in range(Outsourced):
        for k in range(Market):
            for m in range(Products):
                ship_to_market += T_O_MZ[m][l][k]*T2[scen,m,l,k]

    total_outsourcing = b_cost + ship_to_distribution + ship_to_market
    return(total_outsourcing)

def get_rl_rate(w, num_Scenarios, Market, Products):
    rl_penalty = 0
    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                rl_penalty += Probabilities[s]*lost_sales[k][m]*w[s,k,m]*demand[s][m][k]

    return(rl_penalty)

def save_v_values(instance, rl, save_results):
    values = ["v_val_x_i", "v_val_x_j", "v_val_U_km", "v_val_V1_lm", "v_val_V2_lm", "v_val_Q_im", "v_val_Y_ijm", "v_val_Z_jkm", "v_val_T_ljm", "v_val_T_lkm", "v_val_w"]
    v_values = [v_val_x_i, v_val_x_j, v_val_U_km, v_val_V1_lm, v_val_V2_lm, v_val_Q_im, v_val_Y_ijm, v_val_Z_jkm, v_val_T_ljm, v_val_T_lkm, v_val_w]
    if save_results:
        ff = open("/home/dkabe/Model_brainstorming/Output/Variable_vals/" + "Instance_" + str(instance + 1) +  "/variable_vals_" + str(rl) + ".txt", "w+")
        for i in range(len(v_values)):
            if i != len(v_values) - 1:
                ff.write(values[i] + " = " + str(v_values[i]) + '\n')
            else:
                ff.write(values[i] + " = " + str(v_values[i]))
        ff.close()
    return

def PrintToFileSummaryResults():
    results_file = "/home/dkabe/Model_brainstorming/Output/results.txt"
    ff = open(results_file, "a")
    ff.write(str(Summary_dict['ObjVal']) + '\t' + str(Cost_dict['f1']) + '\t' + str(Cost_dict['f2']) + '\t' + str(Cost_dict['f3']) + '\t' + str(Cost_dict['f4']) + '\t')
    ff.write(str(Summary_dict['Demand_met']) + '\t' + str(Summary_dict['Demand_outsourced']) + '\t')
    ff.write(str(Summary_dict['OpenMPs']) + '\t' + str(Summary_dict['OpenDCs']) + '\t')
    ff.write(str(Summary_dict['CPU']))
    ff.write('\n')
    ff.close()
    return


def run_Model(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, objDict, save_results=0):
    for key, value in objDict.items():
        objWeights[key] = value

    InitializeModelParams(instance, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced)
    SetGurobiModel(rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon)
    SolveModel(num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced)
    PrintToFileSummaryResults()
    save_v_values(instance, rl, save_results)
