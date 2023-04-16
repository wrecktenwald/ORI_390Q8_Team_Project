"""
Implement the model detailed in theory in Team Project Proposal Follow-up v2.docx
"""
# %%
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pyomo.kernel as pmo
import pyomo.environ as pyo
import itertools

from ercot_prices import ercot_dam_prices, ercot_rtm_prices


dam_2022 = ercot_dam_prices(year=2022)  # note 2020 is a leap year

lz_lcra_dam_2022 = dam_2022.loc[dam_2022.SETTLEMENT_POINT == 'LZ_LCRA'][['YEAR', 'MONTH', 'DAY', 'HOUR_ENDING', 'PRICE']]
lz_lcra_dam_2022.sort_values(['YEAR', 'MONTH', 'DAY', 'HOUR_ENDING'], inplace=True)
lz_lcra_dam_2022.reset_index(inplace=True, drop=True)
# %%
# hours = 8760
hours = 24 * 7  
# hours = 24 # testing for limited set of hours
batch_runtime = 2 # number of hours it take to charge the battery (no discharge allowed)

# Setting up in MW/MWh
D_t = 0.9 * 100  # OR = (1 - 0.05) * 100
C_t = 0.95 * 100
S_0 = 100
# TODO: leap years in theta_t
theta_t = np.linspace(1, 0.8, hours * 10)  # decline to 80% after ten years
theta_t = np.concatenate([theta_t, np.linspace(theta_t[-1], 0.1, hours * 10)])  # decline to 10% after twenty years, should be non-linear, concave
theta_t = np.concatenate([theta_t, np.linspace(theta_t[-1], 0.0, hours * 10)])  # decline to 0% after thirty years, should be non-linear, convex
S_t = S_0 * theta_t
p_t = lz_lcra_dam_2022['PRICE'].to_numpy()  # TODO: process for node to time series indexed by t
v__c = 0
v__d = 0
R_ij = 0.915  # TODO linear interp between years, get matrix defined for valid ij pairs
r = 0.05  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
T = np.arange(0, hours)  # use native 0 indexing in Python

big_M = float('inf')
max_charge = 2 # maximum number of batch charges allowed

# %%
# LP
model = ConcreteModel()
model.constraints = ConstraintList()
model.d = Var(T, domain=NonNegativeReals) # Discharge
model.c = Var(T, domain=NonNegativeReals) # Charge
model.s = Var(T, domain=NonNegativeReals) # Current storage
model.charging = Var(T, domain=Binary) # Binary variable indicating whether batch charging is in progress


# TODO: use R_ij matrix
model.objective = Objective(
    expr=sum(((p_t[t] - v__d) * R_ij * model.d[t] - (p_t[t] + v__c) * model.c[t]) * np.exp(-r * t / 8760) for t in T),  
    sense=maximize
)

# TODO: can these be vectorized, otherwise streamlined
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
    
    model.constraints.add(model.d[period] <= big_M * (1 - model.charging[period])) # cannot discharge while charging


# solver_factory = SolverFactory('glpk')
solver_factory = SolverFactory('gurobi')
results = solver_factory.solve(model)

print(f"""
Solution Results for model
    STATUS: {results['Solver'][0]['Status'].value.upper()}
    TERMINATION CONDITION: {results['Solver'][0]['Termination condition'].value.upper()}
    OBJECTIVE LOWER BOUND: {results['Problem'][0]['Lower bound']}
    OBJECTIVE UPPER BOUND: {results['Problem'][0]['Lower bound']}
""")
# %%

model.solutions.store_to(results)
results.write(filename='proposal_follow_up_v2_model.json', format='json')

# %%
