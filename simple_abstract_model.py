"""
Model setup:

min SUM(j=0..n) c[j]*x[j]
such that:
1) SUM(j=0..n) a[i,j]*x[j] >= b[i] for every i in (0..m)
2) x[j] >= 0 for every j in (0..n)

Adapted from code shared by Minh Vu
"""

# used to ensure that int or long division arguments are converted to floating point values before division is performed
from __future__ import division

from pyomo.environ import *
from pyomo.opt import SolverFactory

# define model
model = AbstractModel(name = 'simple abstract model')

# define scalers
m = 1
n = 2

# define sets that parematers are defined over
model.i                 = Set(initialize = [l for l in range(m)], ordered=True)
model.j                 = Set(initialize = [l for l in range(n)], ordered=True)

# define model parameters
model.a                 = Param(model.i, model.j)
model.b                 = Param(model.i)
model.c                 = Param(model.j)

## load data into parameters
data = DataPortal()
data.load(filename = 'data/opt_model_data/a.csv', param = model.a, format='array')
data.load(filename = 'data/opt_model_data/b.csv', select = ('i', 'b'), param = model.b, index = model.i)
data.load(filename = 'data/opt_model_data/c.csv', select = ('j', 'c'), param = model.c, index = model.j)

# define variables
model.x                 = Var(model.j, domain=NonNegativeReals)

def obj_expression(model):
    # return summation(model.c, model.x)
    return sum(model.c[j] * model.x[j] for j in model.j)
model.OBJ = Objective(rule=obj_expression)

def ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    return sum(model.a[i,j] * model.x[j] for j in model.j) >= model.b[i]
# the next line creates one constraint FOR EVERY member of the set model.i
model.AxbConstraint = Constraint(model.i, rule=ax_constraint_rule)

# create instance of the model (abstract only)
model = model.create_instance(data)

# solve the model
opt = SolverFactory('glpk')
status = opt.solve(model) 

# write model outputs to a JSON file
model.solutions.store_to(status)
status.write(filename='simple_abstract_results.json', format='json')

# solve command
# pyomo solve simple_abstract_model.py --solver=glpk