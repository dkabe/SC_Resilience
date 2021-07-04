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
V1_lm = {} # quantity of products purchased from outsourcing (epsilon)
V2_lm = {} # quantity of products purchased from outsourcing (past epsilon)
Q_im = {} # quantity produced
Y_ijm = {} # shipping i -> j
Z_jkm = {} # shipping j -> k
T_ljm = {} # shipping l -> j
T_lkm = {} # shipping l -> k
y_lm = {} # indicator variable for step function

first_stage_decisions = {}

grbModel_det = Model('deterministic')

def SetGurobiModel_det(s1):

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
            for j in range(Distribution):
                T_ljm[m,l,j] = grbModel_det.addVar(vtype = GRB.CONTINUOUS)

    for m in range(Products):
        for l in range(Outsourced):
            for k in range(Market):
                    T_lkm[m,l,k] = grbModel_det.addVar(vtype = GRB.CONTINUOUS)

    for m in range(Products):
        for l in range(Outsourced):
            y_lm[m,l] = grbModel_det.addVar(vtype = GRB.BINARY)

    SetGrb_Obj_det()
    ModelCons_det(s1)

def SolveModel_det(s1):
    grbModel_det.params.OutputFlag = 0
    grbModel_det.optimize()
    v_val_x_i = grbModel_det.getAttr('x', x_i)
    v_val_x_j = grbModel_det.getAttr('x', x_j)
    first_stage_decisions[str(s1) + "_" + "x_i"] = v_val_x_i
    first_stage_decisions[str(s1) + "_" + "x_j"] = v_val_x_j
    return

# Objective

def SetGrb_Obj_det():

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

    ship_1 = 0
    ship_2 = 0
    ship_3 = 0
    ship_4 = 0
    for i in range(Manufacturing_plants):
        for j in range(Distribution):
            for m in range(Products):
                ship_1 += Transportation_i_j[m][i][j]*Y_ijm[m,i,j]

    for j in range(Distribution):
        for k in range(Market):
            for m in range(Products):
                ship_2 += Transportation_j_k[m][j][k]*Z_jkm[m,j,k]

    for l in range(Outsourced):
        for j in range(Distribution):
            for m in range(Products):
                ship_3 += T_O_DC[m][l][j]*T_ljm[m,l,j]

    for l in range(Outsourced):
        for k in range(Market):
            for m in range(Products):
                ship_4 += T_O_MZ[m][l][k]*T_lkm[m,l,k]

    total_shipment += ship_1 + ship_2 + ship_3 + ship_4

    # Production
    pr_cost = 0
    for i in range(Manufacturing_plants):
        for m in range(Products):
            pr_cost += Manufacturing_costs[i][m]*Q_im[m,i]

    total_pr_cost += pr_cost

    # Buying from outsource cost
    b_cost = 0
    for l in range(Outsourced):
        for m in range(Products):
            b_cost += Supplier_cost[0][m][l]*V1_lm[m,l] + Supplier_cost[1][m][l]*V2_lm[m,l]
    total_b_cost += b_cost

    #Lost Sales
    l_cost = 0
    for k in range(Market):
        for m in range(Products):
            l_cost += lost_sales[k][m]*U_km[k,m]

    total_l_cost += l_cost

    grb_expr += OC_1 + OC_2 + (total_shipment + total_pr_cost + total_b_cost + total_l_cost)

    grbModel_det.setObjective(grb_expr, GRB.MINIMIZE)

    return

def ModelCons_det(s1):

    # Network Flow

    grbModel_det.addConstrs(Q_im[m,i] >= quicksum(Y_ijm[m,i,j] for j in range(Distribution))
                         for i in range(Manufacturing_plants) for m in range(Products))

    grbModel_det.addConstrs((quicksum(Y_ijm[m,i,j] for i in range(Manufacturing_plants)) +
                         quicksum(T_ljm[m,l,j] for l in range(Outsourced))) >= quicksum(Z_jkm[m,j,k] for k in range(Market))
                        for j in range(Distribution) for m in range(Products))

    grbModel_det.addConstrs((quicksum(Z_jkm[m,j,k] for j in range(Distribution)) +
                         quicksum(T_lkm[m,l,k] for l in range(Outsourced)) + U_km[k,m]) >= demand[s1][m][k]
                         for k in range(Market) for m in range(Products))


    # Purchasing Constraints (everything purchased from outsourced facilities must be shipped)
    grbModel_det.addConstrs(V1_lm[m,l] + V2_lm[m,l] >= quicksum(T_ljm[m,l,j] for j in range(Distribution)) +
                        quicksum(T_lkm[m,l,k] for k in range(Market))
                        for m in range(Products) for l in range(Outsourced))

    # Capacity Constraints
    grbModel_det.addConstrs(quicksum(volume[m]*Q_im[m,i] for m in range(Products)) <= Scenarios[s1][0][i]*Capacities_i[i]*x_i[i]
                         for i in range(Manufacturing_plants))

    grbModel_det.addConstrs(quicksum(volume[m]*Y_ijm[m,i,j] for i in range(Manufacturing_plants) for m in range(Products)) +
                        quicksum(volume[m]*T_ljm[m,l,j] for l in range(Outsourced) for m in range(Products)) <=
                        Scenarios[s1][1][j]*Capacities_j[j]*x_j[j]
                        for j in range(Distribution))

    grbModel_det.addConstrs((V1_lm[m,l] + V2_lm[m,l] <= (Capacities_l[m][l]))
                        for l in range(Outsourced) for m in range(Products))


    # Indicator variable constraints for step function (25 is arbitrary)
    grbModel_det.addConstrs(V1_lm[m,l] <= epsilon
                               for m in range(Products) for l in range(Outsourced))
    return

def run_Model_det(s1):

    SetGurobiModel_det(s1)
    SolveModel_det(s1) 
