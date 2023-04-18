"""
Sensitivity analysis for long term duration storage modeleled at a weekly level

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

wks = 52  # weekly, run one year in circular model
hrs_p = dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])['PRICE'].to_numpy()
wks_p_ids = np.arange(len(hrs_p)) // (7 * 24)
wks_p = np.bincount(wks_p_ids, hrs_p) / np.bincount(wks_p_ids)

mdl = vp_v4_0(
    valid_pair_span=wks,
    periods=wks,
    D=np.full(wks, 100 / 21),  # continue to use a normalized size
    C=np.full(wks, 100 / 31),  # continue to use a normalized size
    S=np.full(wks, 100),  # no system degradation
    p=wks_p[:wks],
    r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
    R=0.95,  # estimate fuel type costs
    batch_c_rt=31,  # attempt to replace natural gas utility type storage with similar seasonality
    batch_d_rt=21,  # attempt to replace natural gas utility type storage with similar seasonality
    vc=0.0,  # estimate costs? 
    vd=0.0,  # estimate costs?
    solver='glpk'
)
mdl.setup_model()
mdl.solve_model()
mdl.write_to_json(filename='model_runs/weekly_long_dur/baseline.json')
