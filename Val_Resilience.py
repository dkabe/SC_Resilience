import math
import time
import numpy as np
import random
from random import randint
from gurobipy import *
import pandas as pd
from random import seed
import matplotlib.pyplot as plt
import itertools
import ast

# Read input files
#path = "C:/Users/Devika Kabe/Documents/Model_brainstorming/Input_Data/"
path = "/home/dkabe/Model_brainstorming/Input_Data/"
p_failure = 0.1
p_running = 1 - p_failure
instances = 6
num_samples = 200
Products  = [2,2,2,2,3,3]

Manufacturing_plants = [2, 3, 4, 6, 6, 6]
Distribution = [3, 4, 6, 8, 4, 4]
Market = [1, 2, 3, 5, 29, 29]
numScenarios = [32, 128, 200, 200, 128, 200]

# Read and append input files
f_i = [None]*instances
f_j = [None]*instances
volume = [None]*instances
Manufacturing_costs = [None]*instances
Transportation_i_j = [None]*instances
Transportation_j_k = [None]*instances
Capacities_i = [None]*instances
Capacities_j = [None]*instances
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

    # Unit cost of lost sales
    lost_sales[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/LostSales_' + str(instance + 1) + '.txt').reshape((Market[instance], Products[instance]))

    # demand
    demand[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/Demand_' + str(instance + 1) + '.txt').reshape((numScenarios[instance], Products[instance], Market[instance]))

    # volume
    volume[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/Volume_' + str(instance + 1) + '.txt')

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

x_i = {} # opening manufacturing plant
x_j = {} # opening DC
U_km = {} # quantity lost sales
Q_im = {} # quantity produced
Y_ijm = {} # shipping i -> j
Z_jkm = {} # shipping j -> k
w_s = {} # penalty for not meeting demand above specified rate

# Dictionaries for analysis
Cost_dict = {}
Summary_dict = {}

# Dictionary to weigh different objectives
objWeights = {}

# Dictionary to save values of each objectives
dic_grbOut = {}

grbModel = Model()

def SetGurobiModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products):

    for i in range(Manufacturing_plants):
        x_i[i] = grbModel.addVar(vtype = GRB.BINARY)

    for j in range(Distribution):
        x_j[j] = grbModel.addVar(vtype = GRB.BINARY)

    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                U_km[s,k,m] = grbModel.addVar(vtype = GRB.CONTINUOUS)    

    for s in range(num_Scenarios):
        for m in range(Products):
            for i in range(Manufacturing_plants):
                Q_im[s,m,i] = grbModel.addVar(vtype = GRB.CONTINUOUS)

    for s in range(num_Scenarios):
        for m in range(Products):
            for i in range(Manufacturing_plants):
                for j in range(Distribution):
                    Y_ijm[s,m,i,j] = grbModel.addVar(vtype = GRB.CONTINUOUS)

    for s in range(num_Scenarios):
        for m in range(Products):
            for j in range(Distribution):
                for k in range(Market):
                    Z_jkm[s,m,j,k] = grbModel.addVar(vtype = GRB.CONTINUOUS)
    
    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                w_s[s,k,m] = grbModel.addVar(vtype = GRB.CONTINUOUS)

    SetGrb_Obj(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products)
    ModelCons(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products)

def SolveModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products):
    grbModel.params.OutputFlag = 0
    grbModel.params.timelimit = 900
    start_time = time.time()
    grbModel.optimize()
    #gap = grbModel.MIPGAP
    # get variable values
    v_val_x_i = grbModel.getAttr('x', x_i)
    v_val_x_j = grbModel.getAttr('x', x_j)
    v_val_U_km = grbModel.getAttr('x', U_km)    
    v_val_Q_im = grbModel.getAttr('x', Q_im)
    v_val_Y_ijm = grbModel.getAttr('x', Y_ijm)
    v_val_Z_jkm = grbModel.getAttr('x', Z_jkm)    
    v_val_w = grbModel.getAttr('x', w_s)

    Summary_dict['ObjVal'] = np.round(grbModel.objval,2)
    Summary_dict["OpenMPs"] = np.sum(v_val_x_i.values())
    Summary_dict["OpenDCs"] = np.sum(v_val_x_j.values())
    Cost_dict["Opening"] =  get_opening_costs(instance, v_val_x_i, v_val_x_j, Manufacturing_plants, Distribution)
    Cost_dict["f4"] = np.round(get_rl_rate(v_val_w, instance, num_Scenarios, Market, Products), 2)

    for s in range(num_Scenarios):
        Summary_dict["Production_" + str(s)] = sum([v_val_Q_im[(s,m,i)] for m in range(Products) for i in range(Manufacturing_plants)])
        Summary_dict["LostSales_" + str(s)] = sum([v_val_U_km[(s,k,m)] for m in range(Products) for k in range(Market)])

    for s in range(num_Scenarios):
        Cost_dict["InHouseShipping_" + str(s)] = get_shipping_costs(instance, s, v_val_Y_ijm, v_val_Z_jkm, Manufacturing_plants, Distribution, Products, Market)
        Cost_dict["Production_" + str(s)] = get_production_cost(instance, s,v_val_Q_im, Manufacturing_plants, Products)
        Cost_dict["LostSales_" + str(s)] = get_lost_cost(instance, s,v_val_U_km, Market, Products)    

    f1_cost = 0
    f3_cost = 0
    for s in range(num_Scenarios):
        f1_cost += Probabilities[instance][s]*(Cost_dict['Production_' + str(s)] + Cost_dict['InHouseShipping_' + str(s)])
        f3_cost +=  Probabilities[instance][s]*Cost_dict['LostSales_' + str(s)]
    Cost_dict["f1"] = np.round(Cost_dict["Opening"] + f1_cost, 2) # in house (opening + production + shipping)
    Cost_dict["f3"] = np.round(f3_cost, 2) # lost sales
    Summary_dict['Demand_met'] = np.sum([Probabilities[instance][s]*(Summary_dict["Production_" + str(s)])/np.sum(demand[instance][s]) for s in range(num_Scenarios)])
    end_time = time.time()

    Summary_dict['CPU'] = end_time - start_time  
    #print("obj val: ", Summary_dict['ObjVal'])
    #print("Opening Decisions: ", sum(v_val_x_i.values()), sum(v_val_x_j.values()))
    #print('In house Cost: ', Cost_dict["f1"])
    #print('Lost Sales: ', Cost_dict["f3"])
    #print('Demand Penalties: ', Cost_dict["f4"])
    #print('Demand being met: ', Summary_dict['Demand_met'])
    #print('Gap: ', gap)
    
    return

# Objective

def SetGrb_Obj(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products):
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
    total_l_cost = 0

    # Shipment

    for s in range(num_Scenarios):
        ship_1 = 0
        ship_2 = 0
        for i in range(Manufacturing_plants):
            for j in range(Distribution):
                for m in range(Products):
                    ship_1 += Transportation_i_j[instance][m][i][j]*Y_ijm[s,m,i,j]

        for j in range(Distribution):
            for k in range(Market):
                for m in range(Products):
                    ship_2 += Transportation_j_k[instance][m][j][k]*Z_jkm[s,m,j,k]
        
        total_shipment += Probabilities[instance][s]*(ship_1 + ship_2)

        # Production
        pr_cost = 0
        for i in range(Manufacturing_plants):
            for m in range(Products):
                pr_cost += Manufacturing_costs[instance][i][m]*Q_im[s,m,i]

        total_pr_cost += Probabilities[instance][s]*pr_cost        

        #Lost Sales
        l_cost = 0
        for k in range(Market):
            for m in range(Products):
                l_cost += lost_sales[instance][k][m]*U_km[s,k,m]

        total_l_cost += Probabilities[instance][s]*l_cost

    # Percentage of demand met
    rl_penalty = 0
    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                rl_penalty += Probabilities[instance][s]*lost_sales[instance][k][m]*w_s[s,k,m]*demand[instance][s][m][k]

    grb_expr += objWeights['f1']*(OC_1 + OC_2 + (total_shipment + total_pr_cost + total_l_cost)) + objWeights['f2']*rl_penalty

    grbModel.setObjective(grb_expr, GRB.MINIMIZE)

    return

    # Model Constraints

def ModelCons(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products):

    # Network Flow

    grbModel.addConstrs(Q_im[s,m,i] >= quicksum(Y_ijm[s,m,i,j] for j in range(Distribution))
                         for s in range(num_Scenarios) for i in range(Manufacturing_plants) for m in range(Products))

    grbModel.addConstrs(quicksum(Y_ijm[s,m,i,j] for i in range(Manufacturing_plants)) >= quicksum(Z_jkm[s,m,j,k] for k in range(Market))
                        for s in range(num_Scenarios) for j in range(Distribution) for m in range(Products))

    grbModel.addConstrs(quicksum(Z_jkm[s,m,j,k] for j in range(Distribution)) +                        
                        U_km[s,k,m] >= demand[instance][s][m][k] for s in range(num_Scenarios) for m in range(Products)
                        for k in range(Market))   

    # Capacity Constraints
    grbModel.addConstrs(quicksum(volume[instance][m]*Q_im[s,m,i] for m in range(Products)) <= Scenarios[instance][s][0][i]*Capacities_i[instance][i]*x_i[i]
                        for s in range(num_Scenarios) for i in range(Manufacturing_plants))

    grbModel.addConstrs(quicksum(volume[instance][m]*Y_ijm[s,m,i,j] for i in range(Manufacturing_plants) for m in range(Products)) <=
                        Scenarios[instance][s][1][j]*Capacities_j[instance][j]*x_j[j] for s in range(num_Scenarios) for s in range(num_Scenarios)
                        for j in range(Distribution)) 

    # Resilience Metric (w = % of rl being missed)
    grbModel.addConstrs(w_s[s,k,m] >= rl - (1 - U_km[s,k,m]/demand[instance][s][m][k]) for s in range(num_Scenarios) for k in range(Market) for m in range(Products))

    return

def get_opening_costs(instance, x1, x2, Manufacturing_plants, Distribution):

    # Cost of opening
    OC_1 = 0
    OC_2 = 0
    for i in range(Manufacturing_plants):
        OC_1 += f_i[instance][i]*x1[i]
    for j in range(Distribution):
        OC_2 += f_j[instance][j]*x2[j]

    Opening = np.round(OC_1 + OC_2)

    return(Opening)

def get_shipping_costs(instance, scen, Y, Z, Manufacturing_plants, Distribution, Products, Market):
    ship_1 = 0
    ship_2 = 0 

    # Shipment
    for i in range(Manufacturing_plants):
        for j in range(Distribution):
            for m in range(Products):
                ship_1 += Transportation_i_j[instance][m][i][j]*Y[scen, m,i,j]

    for j in range(Distribution):
        for k in range(Market):
            for m in range(Products):
                ship_2 += Transportation_j_k[instance][m][j][k]*Z[scen,m,j,k]    

    in_house_shipping = np.round(ship_1 + ship_2)

    return(in_house_shipping)

def get_production_cost(instance, scen, Q, Manufacturing_plants, Products):

    # Production
    pr_cost = 0
    for i in range(Manufacturing_plants):
        for m in range(Products):
            pr_cost += Manufacturing_costs[instance][i][m]*Q[scen,m,i]

    return(np.round(pr_cost))


def get_lost_cost(instance, scen, U, Market, Products):

    #Lost Sales
    l_cost = 0
    for k in range(Market):
        for m in range(Products):
            l_cost += lost_sales[instance][k][m]*U[scen,k,m]

    return(np.round(l_cost))


def get_rl_rate(w, instance, num_Scenarios, Market, Products):
    rl_penalty = 0
    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                rl_penalty += Probabilities[instance][s]*lost_sales[instance][k][m]*w[s,k,m]*demand[instance][s][m][k]

    return(rl_penalty)

def PrintToFileSummaryResults():
    results_file = "/home/dkabe/Model_brainstorming/Output/results2.txt"
    ff = open(results_file, "a")
    ff.write(str(Summary_dict['ObjVal']) + '\t' + str(Cost_dict['f1']) + '\t' + str(Cost_dict['f3']) + '\t' + str(Cost_dict['f4']) + '\t')
    ff.write(str(Summary_dict['Demand_met']) + '\t')
    ff.write(str(Summary_dict['OpenMPs']) + '\t' + str(Summary_dict['OpenDCs']) + '\t')
    ff.write(str(Summary_dict['CPU']))
    ff.write('\n')
    ff.close()
    return


def run_Model(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, objDict):
    for key, value in objDict.items():
        objWeights[key] = value

    SetGurobiModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products)
    SolveModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products)
    PrintToFileSummaryResults()
