import numpy as np
import pandas as pd

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


