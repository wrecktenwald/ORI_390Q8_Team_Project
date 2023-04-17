"""
Run instantiated model from model classes using average ERCOT prices for 2019-2022
"""

import numpy as np
import pandas as pd

from models import proposal_v2_1, linear_model_v1_0


ercot_data_folder = f"{__file__.split('ORI_390Q8_Team_Project')[0]}/ORI_390Q8_Team_Project/data/ERCOT/"
dam_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/dam_avg.csv")
rtm_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/rtm_avg.csv")  # TODO: fix this dataset, use

mod1_hours = 8760
mod1_theta_t = np.linspace(1, 0.8, mod1_hours * 10)  # decline to 80% after ten years
mod1_theta_t = np.concatenate([mod1_theta_t, np.linspace(mod1_theta_t[-1], 0.1, mod1_hours * 10)])  # decline to 10% after twenty years, should be non-linear, concave
mod1_theta_t = np.concatenate([mod1_theta_t, np.linspace(mod1_theta_t[-1], 0.0, mod1_hours * 10)])  # decline to 0% after thirty years, should be non-linear, convex
mod1 = proposal_v2_1(
    valid_pair_span=24,
    periods=mod1_hours,
    D=np.full(mod1_hours, 0.9 * 100),
    C=np.full(mod1_hours, 0.95 * 100),
    S=(100 * mod1_theta_t)[:mod1_hours],
    p=dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])['PRICE'].to_numpy()[:mod1_hours],
    # p=rtm_avg.loc[(rtm_avg['SETTLEMENT_POINT'] == 'LZ_LCRA') & (rtm_avg['TYPE'] == 'LZ')].sort_values(['MONTH', 'DAY', 'HOUR'])['PRICE'].to_numpy()[:mod1_hours],
    r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
    R=0.915, 
    vc=0.0, 
    vd=0.0,
    solver='glpk'
)
mod1.setup_model()
mod1.solve_model()
mod1.write_to_json(filename='mod1.json')

# mod2_hours = 8760
# mod2_theta_t = np.linspace(1, 0.8, mod2_hours * 10)  # decline to 80% after ten years
# mod2_theta_t = np.concatenate([mod2_theta_t, np.linspace(mod2_theta_t[-1], 0.1, mod2_hours * 10)])  # decline to 10% after twenty years, should be non-linear, concave
# mod2_theta_t = np.concatenate([mod2_theta_t, np.linspace(mod2_theta_t[-1], 0.0, mod2_hours * 10)])  # decline to 0% after thirty years, should be non-linear, convex
# mod2 = linear_model_v1_0(
#     max_charge=2,
#     batch_runtime=2, 
#     periods=8760,
#     D=np.full(8760, 0.9 * 100),
#     C=np.full(8760, 0.95 * 100),
#     S=(100 * mod2_theta_t)[:8760],
#     p=dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])['PRICE'].to_numpy(),
#     r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
#     R=0.915, 
#     vc=0.0, 
#     vd=0.0,
#     solver='glpk'
# )
# mod2.setup_model()
# mod2.solve_model()
# mod2.write_to_json(filename='mod2.json')

# Test for daily, weekly granularity tech
