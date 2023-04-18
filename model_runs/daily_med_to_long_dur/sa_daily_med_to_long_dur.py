"""
Sensitivity analysis for medium to long term duration storage modeleled at a daily level

Assumptions for techology TODO based on:
    - ...
"""

import sys
import numpy as np
import pandas as pd

sys.path.append('../ORI_390Q8_Team_Project')

from models import vp_v4_0


ercot_data_folder = f"{__file__.split('ORI_390Q8_Team_Project')[0]}/ORI_390Q8_Team_Project/data/ERCOT/"
dam_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/dam_avg.csv")
rtm_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/rtm_avg.csv")

days = 365  # daily, run one year in circular model
days_p = dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])[['MONTH', 'DAY', 'PRICE']].groupby(
    by=['MONTH', 'DAY'], as_index=False).mean()['PRICE'].to_numpy()

mdl = vp_v4_0(
    valid_pair_span=days,
    periods=days,
    D=np.full(days, 100 / 10),  # continue to use a normalized size
    C=np.full(days, 100 / 20),  # continue to use a normalized size
    S=np.full(days, 100),  # no system degradation
    p=days_p[:days],
    r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
    R=0.95,  # estimate fuel type costs
    batch_c_rt=20,  # attempt to replace natural gas salt type fast cycle storage with similar seasonality
    batch_d_rt=10,  # attempt to replace natural gas salt type fast cycle storage with similar seasonality
    vc=0.0,  # estimate costs? 
    vd=0.0,  # estimate costs?
    solver='glpk'
)
mdl.setup_model()
mdl.solve_model()
mdl.write_to_json(filename='model_runs/daily_med_to_long_dur/baseline.json')
