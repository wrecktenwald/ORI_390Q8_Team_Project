"""
Implement the model detailed in theory in Team Project Proposal Follow-up v2.docx
"""

import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import itertools

from ercot_prices import ercot_dam_prices, ercot_rtm_prices


dam_2022 = ercot_dam_prices(year=2022)  # note 2020 is a leap year

lz_lcra_dam_2022 = dam_2022.loc[dam_2022.SETTLEMENT_POINT == 'LZ_LCRA'][['YEAR', 'MONTH', 'DAY', 'HOUR_ENDING', 'PRICE']]
lz_lcra_dam_2022.sort_values(['YEAR', 'MONTH', 'DAY', 'HOUR_ENDING'], inplace=True)
lz_lcra_dam_2022.reset_index(inplace=True, drop=True)

hours = 8760
# hours = 24 * 7  # testing for limited set of hours

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
valid_pair_span = 24
V = [_ for _ in list(itertools.product(T, T)) if ((_[1] - _[0] <= valid_pair_span) & (_[1] - _[0] > 0))]  # only consider valid periods
# V = [(_, _ + span) for _ in T for span in np.arange(1, valid_pair_span + 1) if _ + span < hours]  # only consider valid periods

# LP
model = ConcreteModel()
model.constraints = ConstraintList()
model.f = Var(V, domain=NonNegativeReals)
model.s = Var(T, domain=NonNegativeReals)

# TODO: use R_ij matrix
model.objective = Objective(
    expr=sum(model.f[_] * ((p_t[_[1]] - v__d) * R_ij - (p_t[_[0]] + v__c)) * np.exp(-r * _[1] / 8760) for _ in V),  
    sense=maximize
)

# TODO: can these be vectorized, otherwise streamlined
for period in T:
    model.constraints.add(model.s[period] <= S_t[period])
    model.constraints.add(model.s[period] == sum(model.f[_] for _ in V if ((_[0] <= period) & (_[1] > period))))
    try: 
        model.constraints.add(sum(model.f[_] for _ in V if (_[1] == period)) <= D_t)
    except ValueError:  # no valid periods
        pass
    try: 
        model.constraints.add(sum(model.f[_] for _ in V if (_[0] == period)) <= C_t)    
    except ValueError:  # no valid periods
        pass

solver_factory = SolverFactory('glpk')
results = solver_factory.solve(model)

print(f"""
Solution Results for model
    STATUS: {results['Solver'][0]['Status'].value.upper()}
    TERMINATION CONDITION: {results['Solver'][0]['Termination condition'].value.upper()}
    OBJECTIVE LOWER BOUND: {results['Problem'][0]['Lower bound']}
    OBJECTIVE UPPER BOUND: {results['Problem'][0]['Lower bound']}
""")

model.solutions.store_to(results)
results.write(filename='proposal_follow_up_v2_model.json', format='json')
