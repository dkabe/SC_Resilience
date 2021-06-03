import math
import numpy as np
import random
from random import randint
from gurobipy import *
import pandas as pd
from random import seed
import matplotlib.pyplot as plt
import itertools

# Read input files
path = "C:/Users/Devika Kabe/Documents/Model_brainstorming/Input_Data/Instance_4/"

Instances = 4
num_samples = 200
num_Scenarios = 200
Manufacturing_plants = 6
Distribution = 8
Market = 5
Products = 2
Outsourced = 2
epsilon = 600
levels = 2

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

Scenarios = []
Manufacturing_plants = [2, 3, 4, 6]
Distribution = [3, 4, 6, 8]

for i in range(Instances):
    a_si = list(itertools.product([1, 0], repeat = Manufacturing_plants[i]))
    a_si = a_si[:-1]
    b_sj = list(itertools.product([1, 0], repeat = Distribution[i]))
    b_sj = b_sj[:-1]
    Scen = [[x,y] for x in a_si for y in b_sj]
    if len(Scen) > num_samples:
        indices = random.sample(range(len(Scen)), num_samples)
        Scen = [Scen[index] for index in indices]
    Scenarios.append(Scen)

# Initialize model variables

x_i = {} # opening manufacturing plant
x_j = {} # opening DC
U_km = {} # quantity lost sales
V1_lm = {} # quantity products purchased from outsourcing below epsilon threshold
V2_lm = {} # quantity products purchased from outsourcing in excess of epsilon threshold
Q_im = {} # quantity produced
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

grbModel = Model('stochasticResil')

def SetGurobiModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced):

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
            for l in range(Outsourced):
                V1_lm[s,m,l] = grbModel.addVar(vtype = GRB.CONTINUOUS)

    for s in range(num_Scenarios):
        for m in range(Products):
            for l in range(Outsourced):
                V2_lm[s,m,l] = grbModel.addVar(vtype = GRB.CONTINUOUS)

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
        for m in range(Products):
            for l in range(Outsourced):
                for j in range(Distribution):
                    T_ljm[s,m,l,j] = grbModel.addVar(vtype = GRB.CONTINUOUS)

    for s in range(num_Scenarios):
        for m in range(Products):
            for l in range(Outsourced):
                for k in range(Market):
                    T_lkm[s,m,l,k] = grbModel.addVar(vtype = GRB.CONTINUOUS)

    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                w_s[s,k,m] = grbModel.addVar(vtype = GRB.CONTINUOUS)

    SetGrb_Obj(num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced)
    ModelCons(rl, instance, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced)

def SolveModel(num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced):
    p_scen = 1/num_Scenarios
    grbModel.params.OutputFlag = 0
    grbModel.optimize()

    # get variable values
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

    obj = grbModel.getObjective()
    print("obj val: ", obj.getValue())

    Summary_dict['ObjVal'] = grbModel.objval
    Summary_dict["OpenMPs"] = np.sum(v_val_x_i.values())
    Summary_dict["OpenDCs"] = np.sum(v_val_x_j.values())
    Cost_dict["Opening"] =  get_opening_costs(v_val_x_i, v_val_x_j, Manufacturing_plants, Distribution)
    Cost_dict["f2"] = np.round(p_scen*get_rl_rate(v_val_w, num_Scenarios, Market, Products))

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

    Purchasing_cost = np.sum([Cost_dict['Purchasing_' + str(s)] for s in range(num_Scenarios)])
    Production_cost = np.sum([Cost_dict['Production_' + str(s)] for s in range(num_Scenarios)])
    LostSales_cost = np.sum([Cost_dict['LostSales_' + str(s)] for s in range(num_Scenarios)])
    InHouseShipping = np.sum([Cost_dict['InHouseShipping_' + str(s)] for s in range(num_Scenarios)])
    OutsourceShipping = np.sum([Cost_dict['OutsourceShipping_' + str(s)] for s in range(num_Scenarios)])

    Cost_dict["f1"] = np.round(Cost_dict["Opening"] + p_scen*(Purchasing_cost + Production_cost + LostSales_cost + InHouseShipping + OutsourceShipping), 2)

    return

# Objective

def SetGrb_Obj(num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced):
    p_scen = 1/num_Scenarios
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

        total_shipment += ship_1 + ship_2 + ship_3 + ship_4

        # Production
        pr_cost = 0
        for i in range(Manufacturing_plants):
            for m in range(Products):
                pr_cost += Manufacturing_costs[i][m]*Q_im[s,m,i]

        total_pr_cost += pr_cost

        # Buying from outsource cost
        b_cost = 0
        for l in range(Outsourced):
            for m in range(Products):
                b_cost += Supplier_cost[0][m][l]*V1_lm[s,m,l] + Supplier_cost[1][m][l]*V2_lm[s,m,l]

        total_b_cost += b_cost

        #Lost Sales
        l_cost = 0
        for k in range(Market):
            for m in range(Products):
                l_cost += lost_sales[k][m]*U_km[s,k,m]

        total_l_cost += l_cost

    # Percentage of demand met
    rl_penalty = 0
    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                rl_penalty += lost_sales[k][m]*w_s[s,k,m]*demand[s][m][k]

    grb_expr += objWeights['f1']*(OC_1 + OC_2 + p_scen*(total_shipment + total_pr_cost + total_b_cost + total_l_cost)) + objWeights['f2']*p_scen*rl_penalty

    grbModel.setObjective(grb_expr, GRB.MINIMIZE)

    return

    # Model Constraints

def ModelCons(rl, instance, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced):

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
    grbModel.addConstrs(quicksum(volume[m]*Q_im[s,m,i] for m in range(Products)) <= Scenarios[instance][s][0][i]*Capacities_i[i]*x_i[i]
                        for s in range(num_Scenarios) for i in range(Manufacturing_plants))

    grbModel.addConstrs(quicksum(volume[m]*Y_ijm[s,m,i,j] for i in range(Manufacturing_plants) for m in range(Products)) +
                        quicksum(volume[m]*T_ljm[s,m,l,j] for l in range(Outsourced) for m in range(Products)) <=
                        Scenarios[instance][s][1][j]*Capacities_j[j]*x_j[j] for s in range(num_Scenarios) for s in range(num_Scenarios)
                        for j in range(Distribution))

    grbModel.addConstrs((V1_lm[s,m,l] + V2_lm[s,m,l] <= (Capacities_l[m][l])) for s in range(num_Scenarios)
                        for l in range(Outsourced) for m in range(Products))


    # Indicator variable constraints for step function (25 is arbitrary)
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

def get_rl_rate(w, num_Scenarios, Market, Products):
    rl_penalty = 0
    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                rl_penalty += lost_sales[k][m]*w[s,k,m]*demand[s][m][k]

    return(rl_penalty)

def run_Model(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, objDict):
    for key, value in objDict.items():
        objWeights[key] = value

    SetGurobiModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced)
    SolveModel(num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced)
