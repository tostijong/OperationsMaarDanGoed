import numpy as np
import pandas as pd
from gurobipy import Model,GRB,LinExpr,quicksum

## Initiate Gurobi model
m = Model()
tab1 = pd.read_csv('tab1.csv', sep=';')
tab2 = pd.read_csv('tab2.csv', sep=';')


### symbol definition
F = tab1['Flight no.'] # flight set
c_fi = tab1['Type'] #size of aircraft which executes flight f_i
L = tab1['Airline'].drop_duplicates() #set of airlines
F_L = {} #TODO: change this in constraints
for airline in L:
    F_L[airline] = tab1.loc[tab1['Airline'] == airline, 'Flight no.']
G = tab2['Gate no.']#Gate set

c_g = tab2['Gate size'] #size of gate g_k
a_fi = tab1['Arr. time'] # arrival time of flight f_i
d_fi = tab1['Dep. time'] # departure time of flight f_i
T = 15 #minimum time interval of two flight which are assigned to the same gate [min]
S_a_gk = tab2['Distance to the baggage hall (unit: m)'] #distance of arrival passenger walkingfrome gate g_k to baggage hall
S_d_gk = tab2['Distance to the security check points (unit: m)'] #distance of departure passenger walking from security to gate g_k
S_m_gk = tab2['Distance to the transit counter (unit: m)'] #distance between gate g_k and transit counter
N_a_fi = tab1['Number of arr. passengers'] #number of arrival passengers of flight f_i
N_d_fi = tab1['Number of dep. passengers'] #number of departure passengers of flight f_i
N_m_fi = tab1['Number of transit passengers'] #number of transit passengers of flight f_i

M = 300 # value of big M
x = 8 # number of contact gates #TODO: nu gehardcode, veranderen als we dat willen

## Decision variables
for i in F.keys():
    for k in G.keys():
        y[i,k] = m.addVar(lb=0, ub=1,
                                vtype=GRB.BINARY,
                                obj = N_a_fi[i]*S_a_gk[k] + N_d_fi[i]*S_d_gk[k] + N_m_fi[i]*S_m_gk[k],
                                name='y[%s,%s]'%(i,k))
for i in F.keys():
    for j in F.keys():
        z[i,j] = m.addVar(lb=0, ub=1,
                                vtype=GRB.BINARY,
                                name='z[%s,%s]'%(i,j))

m.update()
m.setObjective(m.getObjective(), GRB.MINIMIZE)

## Constraints
#C7 - 80% aerobridge
C1 = m.addConstrs((quicksum(quicksum(y[i,k]*(N_a_fi[i] + N_d_fi[i] + N_m_fi[i]) for k in G.keys() if k<x) for i in F.keys())/
                      (quicksum((N_a_fi[i] + N_d_fi[i] + N_m_fi[i]) for i in F.keys())) >= 0.8),
                  name='C1')

#C8 - each flight is assigned to exactly 1 gate
C2 = m.addConstrs((quicksum(y[i,k] == 1 for k in G.keys())
                    for i in F.keys()), name='C2')

#C9 - y is binary (defined in decision variables)

#C10 - z_fi,fj = 1 if fi and fj assigned to same gate
C4 = m.addConstrs((z[i,j] == quicksum(quicksum(quicksum((y[i,k]*y[j,k]) for k in G.keys()) for j in F.keys() if j>i) for i in F.keys())
                  for i in F.keys()
                  for j in F.keys()), name='C4')

#C11 - safety interval if assigned to same gate
C5 = m.addConstrs(((a_fi - d_fi + (1-z[i,j])*M >= T)
                  for i in F.keys()
                  for j in F.keys() if i<j), name='C5')

#C12 - gate type meets AC type
C6 = m.addConstrs((c_fi <= (c_g + (1-y[i,k])*M)
                   for i in F.Keys()
                   for k in G.keys()), name='C6')