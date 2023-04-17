"""
Sample script to show folder
"""
import sys
import numpy as np
import pandas as pd

sys.path.append('../ORI_390Q8_Team_Project')

from models import linear_model_v1_0, vp_v3_0

ercot_data_folder = f"{__file__.split('ORI_390Q8_Team_Project')[0]}/ORI_390Q8_Team_Project/data/ERCOT/"
dam_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/dam_avg.csv")
rtm_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/rtm_avg.csv")

#################################
# Short term sensitivity analysis
#################################
# Baseline parameters (The default parameters before changing ANY single parameter for sensitivity analysis)
# theta_t = np.linspace(1, 0.8, 8760 * 10)

# max_charge=float('inf'),
# batch_runtime=24, 
# periods=hours,
# D=np.full(hours, 0.9 * 100),
# C=np.full(hours, 0.95 * 100),
# S=(100 * theta_t)[:hours],
# p=rtm_avg.loc[rtm_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR'])['PRICE'].to_numpy()[:hours],
# r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
# R=0.915, 
# vc=0.0, 
# vd=0.0,
# # solver='glpk'
# solver='gurobi'

def short_term(input_array):
    hours = 8760 // 2
    theta_t = np.linspace(1, 0.8, 8760 * 10)  # decline to 80% after ten years
    # theta_t = np.concatenate([theta_t, np.linspace(theta_t[-1], 0.1, 8760 * 10)])  # decline to 10% after twenty years, should be non-linear, concave
    # theta_t = np.concatenate([theta_t, np.linspace(theta_t[-1], 0.0, 8760 * 10)])  # decline to 0% after thirty years, should be non-linear, convex
    return [linear_model_v1_0(
        max_charge=float('inf'),
        batch_runtime=24, 
        periods=hours,
        D=np.full(hours, 0.9 * 100),
        C=np.full(hours, 0.95 * 100),
        S=(100 * theta_t)[:hours],
        p=rtm_avg.loc[rtm_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR'])['PRICE'].to_numpy()[:hours],
        r=0.05,  
        R=0.915, 
        vc=value, # <---- the parameter to run sensitivity analysis on
        vd=0.0,
        # solver='glpk' # install through conda install -c conda-forge glpk
        solver='gurobi' # use gurobi if you have it installed instead of glpk
    ).optimal() for value in input_array]


input = range(100, 1001, 250) # array of parameter values to run sens. analysis
res = short_term(input) # this will output an array of optimal objective function values corresponding to the input above