"""
Run instantiated model from model classes using average ERCOT prices for 2019-2022
"""

import numpy as np
import pandas as pd

from models import proposal_v2_1
from models import linear_model_v1_0
from models import vp_v3_0
from models import vp_v4_0
from utilities import read_model_results_json


ercot_data_folder = f"{__file__.split('ORI_390Q8_Team_Project')[0]}/ORI_390Q8_Team_Project/data/ERCOT/"
dam_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/dam_avg.csv")
rtm_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/rtm_avg.csv")

# mod1_hours = 8760
# mod1_theta_t = np.linspace(1, 0.8, mod1_hours * 10)  # decline to 80% after ten years
# mod1_theta_t = np.concatenate([mod1_theta_t, np.linspace(mod1_theta_t[-1], 0.1, mod1_hours * 10)])  # decline to 10% after twenty years, should be non-linear, concave
# mod1_theta_t = np.concatenate([mod1_theta_t, np.linspace(mod1_theta_t[-1], 0.0, mod1_hours * 10)])  # decline to 0% after thirty years, should be non-linear, convex
# mod1 = proposal_v2_1(
#     valid_pair_span=24,
#     periods=mod1_hours,
#     D=np.full(mod1_hours, 0.9 * 100),
#     C=np.full(mod1_hours, 0.95 * 100),
#     S=(100 * mod1_theta_t)[:mod1_hours],
#     # p=dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])['PRICE'].to_numpy()[:mod1_hours],
#     p=rtm_avg.loc[(rtm_avg['SETTLEMENT_POINT'] == 'LZ_LCRA') & (rtm_avg['TYPE'] == 'LZ')].sort_values(['MONTH', 'DAY', 'HOUR'])['PRICE'].to_numpy()[:mod1_hours],
#     r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
#     R=0.915, 
#     vc=0.0, 
#     vd=0.0,
#     solver='glpk'
# )
# mod1.setup_model()
# mod1.solve_model()
# mod1.write_to_json(filename='mod1.json')

# mod2_hours = 8760
# mod2_theta_t = np.linspace(1, 0.8, mod2_hours * 10)  # decline to 80% after ten years
# mod2_theta_t = np.concatenate([mod2_theta_t, np.linspace(mod2_theta_t[-1], 0.1, mod2_hours * 10)])  # decline to 10% after twenty years, should be non-linear, concave
# mod2_theta_t = np.concatenate([mod2_theta_t, np.linspace(mod2_theta_t[-1], 0.0, mod2_hours * 10)])  # decline to 0% after thirty years, should be non-linear, convex
# mod2 = linear_model_v1_0(
#     max_charge=float('inf'),
#     batch_runtime=24, 
#     periods=mod2_hours,
#     D=np.full(mod2_hours, 0.9 * 100),
#     C=np.full(mod2_hours, 0.95 * 100),
#     S=(100 * mod2_theta_t)[:8760],
#     p=dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])['PRICE'].to_numpy()[:mod2_hours],
#     r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
#     R=0.915, 
#     vc=0.0, 
#     vd=0.0,
#     solver='gurobi'
# )
# mod2.setup_model()
# mod2.solve_model()
# mod2.write_to_json(filename='mod2.json')

# mod3_days = 365 * 2  # daily, slow to solve
# mod3_theta_t = np.linspace(1, 0.8, 365 * 10)  # decline to 80% after ten years
# mod3 = proposal_v2_1(
#     valid_pair_span=365,
#     periods=mod3_days,
#     D=np.full(mod3_days, 0.9 * 100),
#     C=np.full(mod3_days, 0.95 * 100),
#     S=(100 * mod3_theta_t)[:mod3_days],
#     p=np.repeat(dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(
#         ['MONTH', 'DAY', 'HOUR_ENDING'])[['MONTH', 'DAY', 'PRICE']].groupby(
#         by=['MONTH', 'DAY'], as_index=False).mean()['PRICE'].to_numpy(), 30)[:mod3_days],
#     r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
#     R=0.88, 
#     vc=0.67, 
#     vd=0.33,
#     solver='glpk'
# )
# mod3.setup_model()
# mod3.solve_model()
# mod3.write_to_json(filename='mod3.json')

# mod4_wks = 52 * 10  # weekly
# mod4_hrs_p = np.repeat(dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])['PRICE'].to_numpy(), 30)
# mod4_wks_p_ids = np.arange(len(mod4_hrs_p)) // (7 * 24)
# mod4_wks_p = np.bincount(mod4_wks_p_ids, mod4_hrs_p) / np.bincount(mod4_wks_p_ids)
# mod4_theta_t = np.linspace(1, 0.8, 52 * 10)  # decline to 80% after ten years
# mod4 = proposal_v2_1(
#     valid_pair_span=104,
#     periods=mod4_wks,
#     D=np.full(mod4_wks, 0.9 * 100),
#     C=np.full(mod4_wks, 0.95 * 100),
#     S=(100 * mod4_theta_t)[:mod4_wks],
#     p=mod4_wks_p[:mod4_wks],
#     r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
#     R=0.88, 
#     vc=0.67, 
#     vd=0.33,
#     solver='glpk'
# )
# mod4.setup_model()
# mod4.solve_model()
# mod4.write_to_json(filename='mod4.json')

mod5_wks = 52 * 30  # weekly
mod5_hrs_p = np.repeat(dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])['PRICE'].to_numpy(), 30)
mod5_wks_p_ids = np.arange(len(mod5_hrs_p)) // (7 * 24)
mod5_wks_p = np.bincount(mod5_wks_p_ids, mod5_hrs_p) / np.bincount(mod5_wks_p_ids)
mod5 = vp_v3_0(
    valid_pair_span=104,
    periods=mod5_wks,
    D=np.full(mod5_wks, 100 / 31),  # continue to use a normalized size
    C=np.full(mod5_wks, 100 / 21),  # continue to use a normalized size
    S=np.full(mod5_wks, 100),  # no system degradation
    p=mod5_wks_p[:mod5_wks],
    r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
    R=0.95,  # estimate fuel type costs, matrix implementation?
    batch_c_rt=21,  # attempt to replace natural gas utility type storage with similar seasonality
    batch_d_rt=31,  # attempt to replace natural gas utility type storage with similar seasonality
    vc=1.5,  # estimate costs? 
    vd=0.0,  # estimate costs?
    solver='glpk'
)
mod5.setup_model()
mod5.solve_model()
mod5.write_to_json(filename='mod5.json')

mod6_wks = 104  # weekly
mod6_hrs_p = np.repeat(dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])['PRICE'].to_numpy(), 2)
mod6_wks_p_ids = np.arange(len(mod6_hrs_p)) // (7 * 24)
mod6_wks_p = np.bincount(mod6_wks_p_ids, mod6_hrs_p) / np.bincount(mod6_wks_p_ids)
mod6 = vp_v4_0(
    valid_pair_span=104,
    periods=mod6_wks,
    D=np.full(mod6_wks, 100 / 31),  # continue to use a normalized size
    C=np.full(mod6_wks, 100 / 21),  # continue to use a normalized size
    S=np.full(mod6_wks, 100),  # no system degradation
    p=mod6_wks_p[:mod6_wks],
    r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
    R=0.95,  # estimate fuel type costs, matrix implementation?
    batch_c_rt=21,  # attempt to replace natural gas utility type storage with similar seasonality
    batch_d_rt=31,  # attempt to replace natural gas utility type storage with similar seasonality
    vc=1.5,  # estimate costs? 
    vd=0.0,  # estimate costs?
    solver='glpk'
)
mod6.setup_model()
mod6.solve_model()
mod6.write_to_json(filename='mod6.json')
