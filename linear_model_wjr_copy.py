"""
Trying to get Minh's model to run on my PC
"""

import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pyomo.kernel as pmo
import pyomo.environ as pyo
import time

from utilities import elapsed_time_display


start_time = time.time()

ercot_data_folder = f"{__file__.split('ORI_390Q8_Team_Project')[0]}/ORI_390Q8_Team_Project/data/ERCOT/"
dam_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/dam_avg.csv")

lz_lcra_dam = dam_avg.loc[dam_avg.SETTLEMENT_POINT == 'LZ_LCRA'][['MONTH', 'DAY', 'HOUR_ENDING', 'PRICE']]
lz_lcra_dam.sort_values(['MONTH', 'DAY', 'HOUR_ENDING'], inplace=True)
lz_lcra_dam.reset_index(inplace=True, drop=True)

# hours = 8760
# hours = 24 * 7  
hours = 24 # testing for limited set of hours
batch_runtime = 2 # number of hours it take to charge the battery (no discharge allowed)

# Setting up in MW/MWh
D_t = 0.9 * 100  # OR = (1 - 0.05) * 100
C_t = 0.95 * 100
S_0 = 100
theta_t = np.linspace(1, 0.8, hours * 10)  # decline to 80% after ten years
theta_t = np.concatenate([theta_t, np.linspace(theta_t[-1], 0.1, hours * 10)])  # decline to 10% after twenty years, should be non-linear, concave
theta_t = np.concatenate([theta_t, np.linspace(theta_t[-1], 0.0, hours * 10)])  # decline to 0% after thirty years, should be non-linear, convex
S_t = S_0 * theta_t
p_t = lz_lcra_dam['PRICE'].to_numpy()
v__c = 0
v__d = 0
R_ij = 0.915  # TODO linear interp between years, get matrix defined for valid ij pairs
r = 0.05  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
T = np.arange(0, hours)  # use native 0 indexing in Python

# big_M = float('inf')
max_charge = 7 # maximum number of batch charges allowed

# LP
model = ConcreteModel()
model.constraints = ConstraintList()
model.d = Var(T, domain=NonNegativeReals) # Discharge
model.c = Var(T, domain=NonNegativeReals) # Charge
model.s = Var(T, domain=NonNegativeReals) # Current storage
model.charging = Var(T, domain=Binary) # Binary variable indicating whether batch charging is in progress

model.objective = Objective(
    expr=sum(((p_t[t] - v__d) * R_ij * model.d[t] - (p_t[t] + v__c) * model.c[t]) * np.exp(-r * t / 8760) for t in T),  
    sense=maximize
)

model.constraints.add(model.s[0] == 0) # initial storage set to 0
model.constraints.add(sum(model.charging[t] for t in T) <= max_charge * batch_runtime)
for period in T:
    model.constraints.add(model.s[period] <= S_t[period])
    model.constraints.add(model.d[period] <= D_t)
    model.constraints.add(model.d[period] <= model.s[period]) # discharge cannot exceed what's currently in storage
    model.constraints.add(model.c[period] <= C_t * model.charging[period]) # if not charging then c[t] is zero

    if period > 0:
        model.constraints.add(model.s[period-1] + model.c[period] - model.d[period] == model.s[period]) # storage equation

        # The following constraints ensure that batch charging lasts exactly batch_runtime
        # Refer to: https://yetanothermathprogrammingconsultant.blogspot.com/2018/03/production-scheduling-minimum-up-time.html
        model.constraints.add(sum([model.charging[k] for k in range(period, min(hours, period + batch_runtime))]) >= batch_runtime * (model.charging[period] - model.charging[period - 1]))   
        model.constraints.add(sum([model.charging[k] for k in range(period, min(hours, period + batch_runtime + 1))]) <= batch_runtime)
    else:
        model.constraints.add(sum([model.charging[k] for k in range(0, min(hours, batch_runtime))]) >= batch_runtime * (model.charging[0]))  
    
    # model.constraints.add(model.d[period] <= big_M * (1 - model.charging[period])) # cannot discharge while charging
    model.constraints.add(model.d[period] <= D_t * (1 - model.charging[period])) # tighter bound, should not reduce feasible region and eliminates big M use

elapsed_time_display(start_time, ' (pyomo model constraint generation)')
start_time = time.time()

solver_factory = SolverFactory('glpk')
results = solver_factory.solve(model)

elapsed_time_display(start_time, ' (pyomo model solve)')
start_time = time.time()

print(f"""
Solution Results for model
    STATUS: {results['Solver'][0]['Status'].value.upper()}
    TERMINATION CONDITION: {results['Solver'][0]['Termination condition'].value.upper()}
    OBJECTIVE LOWER BOUND: {results['Problem'][0]['Lower bound']}
    OBJECTIVE UPPER BOUND: {results['Problem'][0]['Upper bound']}
""")

model.solutions.store_to(results)
results.write(filename='linear_model.json', format='json')
