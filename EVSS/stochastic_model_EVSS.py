import numpy as np
import random
from random import randint
from gurobipy import *
import pandas as pd
from random import seed
import time

np.random.seed(0)

Manufacturing_plants = 2
Distribution = 3
Market = 4
Products = 2
Outsourced = 2
epsilon = 25

# Scenario parameters
a_si = [[1,1], [1,0], [0,1]] # don't include [0,0]
b_sj = [[1,1,1], [1,0,1], [1,1,0], [1,0,0], [0,1,1], [0,1,0], [0,0,1]] # don't include [0,0,0]
Scenarios = [[x,y] for x in a_si for y in b_sj]

num_Scenarios = len(Scenarios)
p_scen = 1/num_Scenarios
# Product Demand
demand = np.random.randint(0,50,(num_Scenarios, Products,Market))

# Cost of opening
f_i = [200, 50]
f_j = [75, 100, 50]

# Unit cost of manufacturing product
Manufacturing_costs = np.random.uniform(0,2, (Manufacturing_plants,Products))

# Unit cost of transporting m from plant to DC
Transportation_i_j = np.random.uniform(0,2, (Products, Manufacturing_plants, Distribution))

# Unit cost of transporting m from DC to Market Zone
Transportation_j_k = np.random.uniform(0,2, (Products, Distribution, Market))

# Plant Capacities: Bigger capacities for the more expensive ones
Capacities_i = np.zeros(Manufacturing_plants) # in volume (metres cubed)
Capacities_i[0] = np.random.randint(800,1000)
Capacities_i[1] = np.random.randint(200,400)
Capacities_j = np.zeros(Distribution) # in volume (metres cubed)
Capacities_j[0] = np.random.randint(400, 600)
Capacities_j[1] = np.random.randint(600, 800)
Capacities_j[2] = np.random.randint(200,400)
Capacities_l = np.random.randint(50,100, (Products,Outsourced)) # in terms of products

# Cost of purchasing product m from supplier l (assume only 1 product type from each outsourcer)
levels = 2
Supplier_cost = np.zeros((levels, Products, Outsourced))
Supplier_cost[0] = np.random.uniform(10,15, (Products, Outsourced))
Supplier_cost[1] = np.random.randint(15,20, (Products, Outsourced))

# Cost of transporting product m from outsourced facility l to j
T_O_DC = np.random.uniform(2, 5, (Products, Outsourced, Distribution))

# Cost of shipping product m from outsourced facility l to k
T_O_MZ = np.random.uniform(5, 7,(Products, Outsourced, Market))

# Product volume
volume = np.random.uniform(2,3,(Products))

# unit cost of lost sales
lost_sales = np.random.randint(18,25,(Market,Products))

# Initialize model variables
x_i = {} # opening manufacturing plant
x_j = {} # opening DC
U_km = {} # quantity lost sales
V1_lm = {} # quantity products purchased from outsourcing (epsilon)
V2_lm = {} # quantity of products purchased from outsourcing after epsilon
Q_im = {} # quantity produced
Y_ijm = {} # shipping i -> j
Z_jkm = {} # shipping j -> k
T_ljm = {} # shipping l -> j
T_lkm = {} # shipping l -> k
y_lm = {} # indicator variable for step function

grbModel = Model('EVSS')

def SetGurobiModel(dict, s1):

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
        for m in range(Products):
            for l in range(Outsourced):
                y_lm[s,m,l] = grbModel.addVar(vtype = GRB.BINARY)


    SetGrb_Obj()
    ModelCons(dict, s1)

def SolveModel():

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
    v_val_y_lm = grbModel.getAttr('x', y_lm)

    obj = grbModel.getObjective()

    return

# Objective

def SetGrb_Obj():

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

    grb_expr += OC_1 + OC_2 + p_scen*(total_shipment + total_pr_cost + total_b_cost + total_l_cost)

    grbModel.setObjective(grb_expr, GRB.MINIMIZE)

    return

# Model Constraints

def ModelCons(dict, s1):

    # Network Flow

    grbModel.addConstrs(Q_im[s,m,i] >= quicksum(Y_ijm[s,m,i,j] for j in range(Distribution))
                         for s in range(num_Scenarios) for i in range(Manufacturing_plants) for m in range(Products))

    grbModel.addConstrs((quicksum(Y_ijm[s,m,i,j] for i in range(Manufacturing_plants)) +
                         quicksum(T_ljm[s,m,l,j] for l in range(Outsourced))) >= quicksum(Z_jkm[s,m,j,k] for k in range(Market))
                        for s in range(num_Scenarios) for j in range(Distribution) for m in range(Products))

    grbModel.addConstrs((quicksum(Z_jkm[s,m,j,k] for j in range(Distribution)) +
                         quicksum(T_lkm[s,m,l,k] for l in range(Outsourced)) + U_km[s,k,m]) >= demand[s][m][k]
                         for s in range(num_Scenarios) for k in range(Market) for m in range(Products))


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

    grbModel.addConstrs(V1_lm[s,m,l] + V2_lm[s,m,l] <= Capacities_l[m][l] for s in range(num_Scenarios)
                        for l in range(Outsourced) for m in range(Products))


    # constraints for step function epsilon
    grbModel.addConstrs(V1_lm[s,m,l] <= epsilon for s in range(num_Scenarios)
                        for l in range(Outsourced) for m in range(Products))

    # constraint to fix opening decision
    grbModel.addConstrs(x_i[i] == dict[str(s1) + "_" + "x_i"][i] for i in range(Manufacturing_plants))
    grbModel.addConstrs(x_j[j] == dict[str(s1) + "_" + "x_j"][j] for j in range(Distribution))

    return

def run_Model(dict, s1):

    SetGurobiModel(dict, s1)
    SolveModel()
