"""
Implement the model detailed in theory in Team Project Proposal Follow-up v2.docx
"""

import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.common.errors import ApplicationError
import time

from utilities import elapsed_time_display


start_time = time.time()

ercot_data_folder = f"{__file__.split('ORI_390Q8_Team_Project')[0]}/ORI_390Q8_Team_Project/data/ERCOT/"
dam_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/dam_avg.csv")

lz_lcra_dam = dam_avg.loc[dam_avg.SETTLEMENT_POINT == 'LZ_LCRA'][['MONTH', 'DAY', 'HOUR_ENDING', 'PRICE']]
lz_lcra_dam.sort_values(['MONTH', 'DAY', 'HOUR_ENDING'], inplace=True)
lz_lcra_dam.reset_index(inplace=True, drop=True)

elapsed_time_display(start_time, ' (ERCOT data loaded)')
start_time = time.time()

hours = 8760
# hours = 24 * 7  # testing for limited set of hours

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
r = 0.05  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
T = np.arange(0, hours)  # use native 0 indexing in Python
valid_pair_span = 24

V = [(_, _ + span) for _ in T for span in np.arange(1, valid_pair_span + 1) if _ + span < hours]  # only consider valid periods

R = np.zeros(shape=(hours, hours), dtype=float)  # TODO: linear interp between years
row_idx = 0  # construct via iteration over limtied distance from matrix diagonal
while row_idx < hours:
    span = 1
    while (span <= valid_pair_span) & (row_idx + span < hours):
        R[row_idx][row_idx + span] = 0.915 # could also be set via rule measuring diff. b/w periods
        span += 1
    row_idx += 1

elapsed_time_display(start_time, ' (V, R generation)')
start_time = time.time()

# LP
model = ConcreteModel()
model.constraints = ConstraintList()
model.f = Var(V, domain=NonNegativeReals)
model.s = Var(T, domain=NonNegativeReals)

model.objective = Objective(
    expr=sum(model.f[idx] * ((p_t[idx[1]] - v__d) * R[idx[0]][idx[1]] - (p_t[idx[0]] + v__c)) * np.exp(-r * idx[1] / 8760) for idx in V),  
    sense=maximize
)

for period in T:  # TODO: can for loop be vectorized, otherwise streamlined
    model.constraints.add(model.s[period] <= S_t[period])
    model.constraints.add(model.s[period] == sum(model.f[idx] for idx in V if ((idx[0] <= period) & (idx[1] > period))))
    try: 
        model.constraints.add(sum(model.f[idx] for idx in V if (idx[1] == period)) <= D_t)
    except ValueError:  # no valid periods
        pass
    try: 
        model.constraints.add(sum(model.f[idx] for idx in V if (idx[0] == period)) <= C_t)    
    except ValueError:  # no valid periods
        pass

elapsed_time_display(start_time, ' (pyomo model constraint generation)')
start_time = time.time()

try:
    solver_factory = SolverFactory('gurobi')
    results = solver_factory.solve(model)
except ApplicationError:  # if gurobi not accessible
    print('Switching to glpk')
    solver_factory = SolverFactory('glpk')
    results = solver_factory.solve(model)

results = solver_factory.solve(model)

elapsed_time_display(start_time, ' (pyomo model solve)')
start_time = time.time()

print(f"""
Solution Results for model
    STATUS: {results['Solver'][0]['Status'].value.upper()}
    TERMINATION CONDITION: {results['Solver'][0]['Termination condition'].value.upper()}
    OBJECTIVE LOWER BOUND: {results['Problem'][0]['Lower bound']}
    OBJECTIVE UPPER BOUND: {results['Problem'][0]['Lower bound']}
""")

model.solutions.store_to(results)
results.write(filename='proposal_follow_up_v2_model.json', format='json')

elapsed_time_display(start_time, ' (pyomo model write)')
start_time = time.time()
