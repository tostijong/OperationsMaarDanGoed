import numpy as np
import pandas as pd
from gurobipy import Model,GRB,LinExpr,quicksum,max_

## Initiate Gurobi model
m = Model()
# tab1 = pd.read_csv('tab1.csv', sep=';')
# tab2 = pd.read_csv('tab2.csv', sep=';')
tab1 = pd.read_excel('big2.xlsx', sheet_name='Flights', nrows = 30)
tab2 = pd.read_excel('big2.xlsx', sheet_name='Gates')
def convert_time_to_minutes(df):
    df_copy = df.copy()
    df_copy= df_copy.apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    return df_copy

### symbol definition
F = tab1['Flight no.'] # flight set
c_fi = tab1['Type'] #size of aircraft which executes flight f_i
c_fi2 = []
for i in c_fi: #erg gebeunde oplossing, maar vgm werkt t wel :) (zelfde geldt voor c_g)
    if i == 'S':
        c_fi2.append(1)
    if i == 'M':
        c_fi2.append(2)
    if i == 'L':
        c_fi2.append(3)
print(c_fi2)
print(c_fi2[0])
print(c_fi)
L = tab1['Airline'].drop_duplicates() #set of airlines
F_L = {airline: tab1.loc[tab1['Airline'] == airline, 'Flight no.'] for airline in L}
G = tab2['Gate no.']#Gate set

c_g = tab2['Gate size'] #size of gate g_k
c_g2 = []
for i in c_g:
    if i == 'S':
        c_g2.append(1)
    if i == 'M':
        c_g2.append(2)
    if i == 'L':
        c_g2.append(3)
print(c_g2)
a_fi = convert_time_to_minutes(tab1['Arr. time']) # arrival time of flight f_i
d_fi = convert_time_to_minutes(tab1['Dep. time']) # departure time of flight f_i
T = 15 #minimum time interval of two flight which are assigned to the same gate [min]
S_a_gk = tab2['Distance to the baggage hall (unit: m)'] #distance of arrival passenger walkingfrome gate g_k to baggage hall
S_d_gk = tab2['Distance to the security check points (unit: m)'] #distance of departure passenger walking from security to gate g_k
S_m_gk = tab2['Distance to the transit counter (unit: m)'] #distance between gate g_k and transit counter
N_a_fi = tab1['Number of arr. passengers'] #number of arrival passengers of flight f_i
N_d_fi = tab1['Number of dep. passengers'] #number of departure passengers of flight f_i
N_m_fi = tab1['Number of transit passengers'] #number of transit passengers of flight f_i
k = tab2['Contact gate or remote stand']
M = 300 # value of big M
x = 0 #Number of contact gates
for type in k:
    if type == 'C':
        x += 1

# x = 8 # number of contact gates #TODO: nu gehardcode, veranderen als we dat willen
Z2 = 0.1 # maximum margin of difference per airline

## Decision variables
y = {}
z = {}

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
C1 = m.addConstr((((quicksum(y[i,k]*(N_a_fi[i] + N_d_fi[i] + N_m_fi[i])
                              for k in G.keys() if k<x
                              for i in F.keys()))
                              / sum([N_a_fi[i] + N_d_fi[i] + N_m_fi[i] for i in F.keys()]))
                              >= 0.8), name='C1')

#C8 - each flight is assigned to exactly 1 gate
C2 = m.addConstrs((quicksum(y[i,k] for k in G.keys()) == 1
                    for i in F.keys()), name='C2')

#C9 - y is binary (defined in decision variables)

#C10 - z_fi,fj = 1 if fi and fj assigned to same gate
# C4 = m.addConstrs((z[i,j] == quicksum(quicksum(quicksum((y[i,k]*y[j,k]) for k in G.keys()) for j in F.keys() if j>i) for i in F.keys())
#                   for i in F.keys()
#                   for j in F.keys() if j>i), name='C4') #zou kunnen dat hier error/fout komt doordat ie loopt over alle j, maar in de quicksum alleen j>i

# C4 = m.addConstrs((z[fi,fj] == quicksum(y[i,k]*y[j,k]
#                                       for i in F.keys()
#                                       for j in F.keys() if j > i
#                                       for k in G.keys())
#                    for fi in F.keys()
#                    for fj in F.keys()),name='C4')
#
C4 = m.addConstrs((z[i,j] == quicksum(y[i,k]*y[j,k]
                                      for k in G.keys())
                   for i in F.keys()
                   for j in F.keys() if j>i),name='C4')

#C11 - safety interval if assigned to same gate
C5 = m.addConstrs(((a_fi[j] - d_fi[i] + (1-z[i,j])*M >= T)
                  for i in F.keys()
                  for j in F.keys() if i<j), name='C5')

#C12 - gate type meets AC type
C6 = m.addConstrs((c_fi2[i] <= (c_g2[k] + (1-y[i,k])*M)
                   for i in F.keys()
                   for k in G.keys()), name='C6')

# C3,4,5,6 - minimum difference per airline
S = m.addVar(vtype=GRB.CONTINUOUS,name='S')
S_val = m.addConstr(S == quicksum((y[i,k]*
            (N_a_fi[i]*S_a_gk[k]+N_d_fi[i]*S_d_gk[k]+N_m_fi[i]*S_m_gk[k])
            for k in G.keys()
            for i in F.keys())) / sum([N_a_fi[i]+N_d_fi[i]+N_m_fi[i] for i in F.keys()]))

S_la = {la: m.addVar(vtype=GRB.CONTINUOUS,name=f'S_la[{la}]') for la in L}
S_la_val = {}
for airline in L:
    S_la_val[airline] = m.addConstr(S_la[airline] == quicksum(y[i,k]*
                              (N_a_fi[i]*
                               S_a_gk[k]+
                               N_d_fi[i]*
                               S_d_gk[k]+
                               N_m_fi[i]*
                               S_m_gk[k])
                              for k in G.keys() for i in F_L[airline].keys())
                     /sum([N_a_fi[i]+N_d_fi[i]+N_m_fi[i] for i in F_L[airline].keys()]), name=f'S_la_val[{airline}]')


# M = 1E6
B = {la: m.addVar(lb=0, ub=1,
                    vtype=GRB.BINARY,
                    name=f'B[{la}]')
                    for la in L}

Z_S_la = {la: m.addVar(vtype=GRB.CONTINUOUS,
                            name=f'Z_S_la[{la}]')
                            for la in L}

C0_max1 = m.addConstrs((Z_S_la[la]*S >= (S_la[la] - S) for la in L),name='C0_max1')
C0_max2 = m.addConstrs((Z_S_la[la]*S >= -1*(S_la[la] - S) for la in L),name='C0_max2')

C0_min1 = m.addConstrs((Z_S_la[la]*S <= (S_la[la] - S) + M*B[la] for la in L),name='C0_min1')
C0_min1 = m.addConstrs((Z_S_la[la]*S <= -1*(S_la[la] - S) + M*(1-B[la]) for la in L),name='C0_min2')



C7 = m.addConstrs(((Z_S_la[la] <= Z2)
                  for la in L), name = 'C7')


# m.setObjective(quicksum(y[i,k]*(N_a_fi[i]*S_a_gk[k] + N_d_fi[i]*S_d_gk[k] + N_m_fi[i]*S_m_gk[k])
#                         for k in G.keys()
#                         for i in F.keys()), GRB.MINIMIZE)




# Z2 = m.addVar(vtype=GRB.CONTINUOUS,name='Z2')
# Z2_val = m.addConstr(Z2 == max_([Z_S_la[la] for la in L]))
# m.setObjective(Z2, GRB.MINIMIZE)

m.setParam('NonConvex', 2)
m.update()
m.optimize()
# m.computeIIS()
# m.write('m_test.ilp')
m.write('operations.lp')


cutoff = 10E-6
GateAssigned = []
for i in F.keys():
    for k in G.keys():
        if y[i,k].X > cutoff:
            print(f'Flight {F[i]} is assigned to gate {G[k]}')
            GateAssigned.append(k)
print(f'S is equal to {S.X}')
for la in L:
    print(f'S_{la} is equal to {S_la[la].X}')
# print(sum([y[i,k].X*(N_a_fi[i]*S_a_gk[k] + N_d_fi[i]*S_d_gk[k] + N_m_fi[i]*S_m_gk[k])
#                         for k in G.keys()
#                         for i in F.keys()]))
# print(S)
# for la in L:
#     print(la,': ',S_la[la])
#     # print(f'{Z_S_la[la].X - np.abs(S_la[la].X - S.X)/S.X}')

# print(quicksum((y[i,k].X*
#             (N_a_fi[i]*S_a_gk[k]+N_d_fi[i]*S_d_gk[k]+N_m_fi[i]*S_m_gk[k])
#             for k in G.keys()
#             for i in F.keys())),'/',sum([N_a_fi[i]+N_d_fi[i]+N_m_fi[i] for i in F.keys()]), '=' ,S)
print(f'Q = {sum([y[i,k].X * (N_a_fi[i] + N_d_fi[i] + N_m_fi[i]) for i in F.keys() for k in G.keys() if k<x]) / sum([N_a_fi[i] + N_d_fi[i] + N_m_fi[i] for i in F.keys()])}')
print("--- Walking distance ---")
print(quicksum((y[i,k].X*
        (N_a_fi[i]*S_a_gk[k]+N_d_fi[i]*S_d_gk[k]+N_m_fi[i]*S_m_gk[k])
        for k in G.keys()
        for i in F.keys())))


import matplotlib.pyplot as plt
xticks_stepsize = 30 #min
ArrT_min = np.array([a_fi[i] for i in F.keys()])
DepT_min = np.array([d_fi[i] for i in F.keys()])

def timestamp(time_min):
    hr = time_min // 60
    min = time_min - hr*60

    hr_str = str(hr)
    min_str = str(min)
    if len(hr_str) == 1:
        hr_str = '0' + hr_str

    if len(min_str) == 1:
        min_str = '0' + min_str

    return hr_str+':'+min_str

Tticks = [t for t in range(int(np.floor(min(ArrT_min)/xticks_stepsize)*xticks_stepsize),
               int((np.ceil(max(DepT_min)/xticks_stepsize)+1)*xticks_stepsize),xticks_stepsize)]
Tticks_str = [timestamp(t) for t in Tticks]

color_mapping = {1: 'green', 2: 'orange', 3: 'red'}
colors = [color_mapping[val] for val in c_fi2]

bars = plt.barh(y=GateAssigned,
                width=DepT_min-ArrT_min,
                left=ArrT_min)
gate_color_mapping = {1: 'green', 2: 'orange', 3: 'red'}
gate_colors = [gate_color_mapping[val] for val in c_g2]

# gate colouring by size, change color=color to color='white' if you don't want it
for i, (color, gate) in enumerate(zip(gate_colors, G)):
    plt.barh(y=gate,
            #  width=max(DepT_min) - min(ArrT_min),  # Set to max width for full background coverage
            #  left=min(ArrT_min),  # Set to min start for full background coverage
             width = Tticks[-1] - Tticks[0],
             left=Tticks[0],
             color=color,
             alpha=0.2,  # High opacity
             edgecolor='none')  # No border

for bar, label in zip(bars, F):
    plt.text(x=bar.get_x() + bar.get_width() / 2,  # x position
             y=bar.get_y() + bar.get_height() / 2,  # y position
             s=label,  # label text
             ha='center',  # horizontal alignment
             va='center',  # vertical alignment
             color='white')  # text color

bars = plt.barh(y=GateAssigned,
                width=DepT_min-ArrT_min,
                left=ArrT_min,
                color = colors,
                alpha=1.0)

plt.xticks(Tticks,Tticks_str)
plt.xlim((Tticks[0],Tticks[-1]))


plt.show()
