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

# Read input files
#path = "C:/Users/Devika Kabe/Documents/Model_brainstorming/Input_Data/"
path = "/home/dkabe/Model_brainstorming/Input_Data/"
p_failure = 0.1
p_running = 1 - p_failure
instances = 6
num_samples = 200
Products  = [2,2,2,2,3,3]
Outsourced =[2,2,2,2,3,3]
#Products = 2
#Outsourced = 2
levels = 2

Manufacturing_plants = [2, 3, 4, 6, 6, 6]
Distribution = [3, 4, 6, 8, 4, 4]
Market = [1, 2, 3, 5, 29, 29]
numScenarios = [32, 128, 200, 200, 200, 200]

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
compatibility = [None]*instances

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

    # Compatibility of plants to products
    compatibility[instance] = np.loadtxt(path + 'Instance_' + str(instance + 1) + '/Compatibilities_' + str(instance + 1) + '.txt')


Scenarios = []
Probabilities = []

for i in range(instances):
    a_si = list(itertools.product([1, 0], repeat = Manufacturing_plants[i]))
    b_sj = list(itertools.product([1, 0], repeat = Distribution[i]))
    Scen = [[x,y] for x in a_si for y in b_sj]
    p_scen = []
    for s in range(len(Scen)):
        p_i = (p_running**(np.sum(Scen[s][0]) + np.sum(Scen[s][1])))*(p_failure**(Manufacturing_plants[i] + Distribution[i] - (np.sum(Scen[s][0]) + np.sum(Scen[s][1]))))
        p_scen.append(p_i)
    if len(Scen) > num_samples:
        random.seed(0)
        indices = random.sample(range(len(Scen)), num_samples)
        Scen = [Scen[index] for index in indices]
        p_scen = [p_scen[index] for index in indices]
        p_factor = 1/np.sum(p_scen)
        p_scen = list(map(lambda x: x*p_factor, p_scen))
    Scenarios.append(Scen)
    Probabilities.append(p_scen)

#np.savetxt("scen.txt", Scenarios[1], fmt='%s')
# Initialize model variables

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

grbModel = Model('stochasticResil')

def SetGurobiModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon):

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

    SetGrb_Obj(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced)
    ModelCons(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon)

def SolveModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced):
    grbModel.params.OutputFlag = 0
    grbModel.params.timelimit = 900
    start_time = time.time()
    grbModel.optimize()
    #gap = grbModel.MIPGAP
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

    Summary_dict['ObjVal'] = np.round(grbModel.objval,2)
    Summary_dict["OpenMPs"] = np.sum(v_val_x_i.values())
    Summary_dict["OpenDCs"] = np.sum(v_val_x_j.values())
    Cost_dict["Opening"] =  get_opening_costs(instance, v_val_x_i, v_val_x_j, Manufacturing_plants, Distribution)
    Cost_dict["f4"] = np.round(get_rl_rate(v_val_w, instance, num_Scenarios, Market, Products), 2)

    for s in range(num_Scenarios):
        Summary_dict["Purchasing_" + str(s)] = sum([v_val_V1_lm[(s,m,l)] +  v_val_V2_lm[(s,m,l)] for m in range(Products) for l in range(Outsourced)])
        Summary_dict["Production_" + str(s)] = sum([v_val_Q_im[(s,m,i)] for m in range(Products) for i in range(Manufacturing_plants)])
        Summary_dict["LostSales_" + str(s)] = sum([v_val_U_km[(s,k,m)] for m in range(Products) for k in range(Market)])
        Summary_dict["OutsourceToDC_" + str(s)] = sum([v_val_T_ljm[(s,m,l,j)] for m in range(Products) for l in range(Outsourced) for j in range(Distribution)])
        Summary_dict["OutsourceToMarket_" + str(s)] = sum([v_val_T_lkm[(s,m,l,k)] for m in range(Products) for l in range(Outsourced) for k in range(Market)])

    for s in range(num_Scenarios):
        Cost_dict["InHouseShipping_" + str(s)] = get_shipping_costs(instance, s,v_val_Y_ijm, v_val_Z_jkm, v_val_T_ljm, v_val_T_lkm, Manufacturing_plants, Distribution, Products, Market, Outsourced)[0]
        Cost_dict["OutsourceShipping_" + str(s)] = get_shipping_costs(instance, s,v_val_Y_ijm, v_val_Z_jkm, v_val_T_ljm, v_val_T_lkm, Manufacturing_plants, Distribution, Products, Market, Outsourced)[1]
        Cost_dict["Production_" + str(s)] = get_production_cost(instance, s,v_val_Q_im, Manufacturing_plants, Products)
        Cost_dict["Purchasing_" + str(s)] = get_purchase_costs(instance, s,v_val_V1_lm, v_val_V2_lm, Outsourced, Products)
        Cost_dict["LostSales_" + str(s)] = get_lost_cost(instance, s,v_val_U_km, Market, Products)    

    f1_cost = 0
    f2_cost = 0
    f3_cost = 0
    for s in range(num_Scenarios):
        f1_cost += Probabilities[instance][s]*(Cost_dict['Production_' + str(s)] + Cost_dict['InHouseShipping_' + str(s)])
        f2_cost += Probabilities[instance][s]*(Cost_dict['Purchasing_' + str(s)] + Cost_dict['OutsourceShipping_' + str(s)])
        f3_cost +=  Probabilities[instance][s]*Cost_dict['LostSales_' + str(s)]
    Cost_dict["f1"] = np.round(Cost_dict["Opening"] + f1_cost, 2) # in house (opening + production + shipping)
    Cost_dict["f2"] = np.round(f2_cost, 2) # Outsourcing # (purchasing + shipping)
    Cost_dict["f3"] = np.round(f3_cost, 2) # lost sales
    Summary_dict['Demand_met'] = np.sum([Probabilities[instance][s]*(Summary_dict["Purchasing_" + str(s)] + Summary_dict["Production_" + str(s)])/np.sum(demand[instance][s]) for s in range(num_Scenarios)])
    Summary_dict['Demand_outsourced'] = np.sum([Probabilities[instance][s]*Summary_dict["Purchasing_" + str(s)]/np.sum(demand[instance][s]) for s in range(num_Scenarios)])
    end_time = time.time()

    Summary_dict['CPU'] = end_time - start_time  
    #print("obj val: ", Summary_dict['ObjVal'])
    #print("Opening Decisions: ", sum(v_val_x_i.values()), sum(v_val_x_j.values()))
    #print('In house Cost: ', Cost_dict["f1"])
    #print('Outsource Cost: ', Cost_dict["f2"])
    #print('Lost Sales: ', Cost_dict["f3"])
    #print('Demand Penalties: ', Cost_dict["f4"])
    #print('Unweighted Demand purchased from outsourcing: ', np.mean([Summary_dict["Purchasing_" + str(s)]/np.sum(demand[instance][s]) for s in range(num_Scenarios)]))
    #print('Demand being met: ', Summary_dict['Demand_met'])
    #print('Weighted Demand purchased from outsourcing: ', Summary_dict['Demand_outsourced'])
    #print('Gap: ', gap)

    
    return

# Objective

def SetGrb_Obj(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced):
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

    # Percentage of demand met
    rl_penalty = 0
    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                rl_penalty += Probabilities[instance][s]*lost_sales[instance][k][m]*w_s[s,k,m]*demand[instance][s][m][k]

    grb_expr += objWeights['f1']*(OC_1 + OC_2 + (total_shipment + total_pr_cost + total_b_cost + total_l_cost)) + objWeights['f2']*rl_penalty

    grbModel.setObjective(grb_expr, GRB.MINIMIZE)

    return

    # Model Constraints

def ModelCons(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon):

    # Network Flow

    grbModel.addConstrs(Q_im[s,m,i] >= quicksum(Y_ijm[s,m,i,j] for j in range(Distribution))
                         for s in range(num_Scenarios) for i in range(Manufacturing_plants) for m in range(Products))

    grbModel.addConstrs((quicksum(Y_ijm[s,m,i,j] for i in range(Manufacturing_plants)) +
                         quicksum(T_ljm[s,m,l,j] for l in range(Outsourced))) >= quicksum(Z_jkm[s,m,j,k] for k in range(Market))
                        for s in range(num_Scenarios) for j in range(Distribution) for m in range(Products))

    grbModel.addConstrs(quicksum(Z_jkm[s,m,j,k] for j in range(Distribution)) +
                        quicksum(T_lkm[s,m,l,k] for l in range(Outsourced)) +
                        U_km[s,k,m] >= demand[instance][s][m][k] for s in range(num_Scenarios) for m in range(Products)
                        for k in range(Market))


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

    grbModel.addConstrs((V1_lm[s,m,l] + V2_lm[s,m,l] <= (Capacities_l[instance][m][l])) for s in range(num_Scenarios)
                        for l in range(Outsourced) for m in range(Products))
    
    # Compatibility Constraints
    #grbModel.addConstrs(Q_im[s,m,i]  <= Scenarios[instance][s][0][i]*Capacities_i[instance][i]*x_i[i]*compatibility[instance][m][i]
     #                   for m in range(Products) for s in range(num_Scenarios) for i in range(Manufacturing_plants))


    # Indicator variable constraints for step function 
    grbModel.addConstrs(V1_lm[s,m,l] <= epsilon for s in range(num_Scenarios) for m in range(Products) for l in range(Outsourced))


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

def get_shipping_costs(instance, scen, Y, Z, T1, T2, Manufacturing_plants, Distribution, Products, Market, Outsourced):
    ship_1 = 0
    ship_2 = 0
    ship_3 = 0
    ship_4 = 0

    # Shipment
    for i in range(Manufacturing_plants):
        for j in range(Distribution):
            for m in range(Products):
                ship_1 += Transportation_i_j[instance][m][i][j]*Y[scen, m,i,j]

    for j in range(Distribution):
        for k in range(Market):
            for m in range(Products):
                ship_2 += Transportation_j_k[instance][m][j][k]*Z[scen,m,j,k]

    for l in range(Outsourced):
        for j in range(Distribution):
            for m in range(Products):
                ship_3 += T_O_DC[instance][m][l][j]*T1[scen,m,l,j]

    for l in range(Outsourced):
        for k in range(Market):
            for m in range(Products):
                ship_4 += T_O_MZ[instance][m][l][k]*T2[scen,m,l,k]

    in_house_shipping = np.round(ship_1 + ship_2)

    outsourced_shipping = np.round(ship_3 + ship_4)

    return(in_house_shipping, outsourced_shipping)

def get_production_cost(instance, scen, Q, Manufacturing_plants, Products):

    # Production
    pr_cost = 0
    for i in range(Manufacturing_plants):
        for m in range(Products):
            pr_cost += Manufacturing_costs[instance][i][m]*Q[scen,m,i]

    return(np.round(pr_cost))

def get_purchase_costs(instance, scen, V1, V2, Outsourced, Products):

    # Buying from outsource cost
    b_cost = 0
    for l in range(Outsourced):
        for m in range(Products):
            b_cost += Supplier_cost[instance][0][m][l]*V1[scen,m,l] + Supplier_cost[instance][1][m][l]*V2[scen,m,l]

    return(np.round(b_cost))

def get_lost_cost(instance, scen, U, Market, Products):

    #Lost Sales
    l_cost = 0
    for k in range(Market):
        for m in range(Products):
            l_cost += lost_sales[instance][k][m]*U[scen,k,m]

    return(np.round(l_cost))

def get_outsourced_cost(instance, scen, V1, V2, T1, T2, Distribution, Products, Outsourced, Market):
    # Buying from outsource cost
    b_cost = 0
    ship_to_distribution = 0
    ship_to_market = 0
    for l in range(Outsourced):
        for m in range(Products):
            b_cost += Supplier_cost[instance][0][m][l]*V1[scen,m,l] + Supplier_cost[instance][1][m][l]*V2[scen,m,l]

    # Shipping from outsourced cost
    for l in range(Outsourced):
        for j in range(Distribution):
            for m in range(Products):
                ship_to_distribution += T_O_DC[instance][m][l][j]*T1[scen,m,l,j]

    for l in range(Outsourced):
        for k in range(Market):
            for m in range(Products):
                ship_to_market += T_O_MZ[instance][m][l][k]*T2[scen,m,l,k]

    total_outsourcing = b_cost + ship_to_distribution + ship_to_market
    return(total_outsourcing)

def get_rl_rate(w, instance, num_Scenarios, Market, Products):
    rl_penalty = 0
    for s in range(num_Scenarios):
        for k in range(Market):
            for m in range(Products):
                rl_penalty += Probabilities[instance][s]*lost_sales[instance][k][m]*w[s,k,m]*demand[instance][s][m][k]

    return(rl_penalty)

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


def run_Model(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon, objDict):
    for key, value in objDict.items():
        objWeights[key] = value

    SetGurobiModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced, epsilon)
    SolveModel(instance, rl, num_Scenarios, Manufacturing_plants, Distribution, Market, Products, Outsourced)
    PrintToFileSummaryResults()
