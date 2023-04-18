"""
Sensitivity analysis for Liquid Air Energy Storage (LAES) applied to daily periods
"""

import sys
import numpy as np
import pandas as pd

sys.path.append('../ORI_390Q8_Team_Project')

from models import vp_v4_0


ercot_data_folder = f"{__file__.split('ORI_390Q8_Team_Project')[0]}/ORI_390Q8_Team_Project/data/ERCOT/"
dam_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/dam_avg.csv")
rtm_avg = pd.read_csv(f"{ercot_data_folder}/post_processing/rtm_avg.csv")

try:
    sa_output = pd.read_csv('model_runs/laes/laes.csv')  # update during runs 
except FileNotFoundError:
    print('Warning will create .csv')
    sa_output = pd.DataFrame()  # fill during runs

days = 365  # daily, run one year in circular model
days_p = dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == 'LZ_LCRA'].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])[['MONTH', 'DAY', 'PRICE']].groupby(
    by=['MONTH', 'DAY'], as_index=False).mean()['PRICE'].to_numpy()  # TODO: switch pricing zone to near ATX, rtm? 

big_S = 1  # normalize to 1 MWh
baseline = dict(
    R=0.675,  # 2020 tech estimated 60-75%
    valid_pair_span=days, 
    DadnC=0.2
)
sa_params = dict(
    R=np.arange(0.6, 1 + 0.05, 0.01),
    valid_pair_span=np.concatenate((np.arange(1, 10 + 1, 1), np.arange(15, 60 + 5, 5))),
    DadnC=np.concatenate((np.arange(0.01, 0.1 + 0.2, 0.2), np.arange(0.2, 1.0 + 0.1, 0.1)))
)

mdl_params = baseline.copy()
mdl = vp_v4_0(
    periods=days,
    S=np.full(days, big_S),  # no system degradation
    p=days_p[:days],
    r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
    solver='glpk',
    D=np.full(days, mdl_params['DandC']),
    C=np.full(days, mdl_params['DandC']),
    **{k: v for k, v in mdl_params.items() if k not in ['D', 'C', 'DandC']}
)
mdl.setup_model()
mdl.solve_model()
sa_output = pd.concat([sa_output, pd.DataFrame.from_dict({
    'SA Param': np.array(['Baseline']),
    **{k: np.array([v]) for k, v in mdl_params.items()},
    'Lower bound': np.array([mdl.results['Problem'][0]['Lower bound']]),
    'Upper bound': np.array([mdl.results['Problem'][0]['Upper bound']]),
    'Optimality Gap': np.array([mdl.results['Solution'][0]['Gap']]),
    'Status': np.array([mdl.results['Solution'][0]['Status']]),
    'Objective': np.array([mdl.results['Solution'][0]['Objective']['objective']['Value']])
})], ignore_index=True)
print('Ran SA for baseline')
mdl.write_to_json(filename='model_runs/daily_med_to_long_dur/baseline.json')

for par, vals in sa_params.items():  # iterate over models editing baseline by param
    for val in vals:
        mdl_params = baseline.copy()
        mdl_params.update({par: val})
        mdl = vp_v4_0(
            valid_pair_span=days,
            periods=days,
            S=np.full(days, big_S),  # no system degradation
            p=days_p[:days],
            r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
            solver='glpk',
            D=np.full(days, mdl_params['DandC']),
            C=np.full(days, mdl_params['DandC']),
            **{k: v for k, v in mdl_params.items() if k not in ['D', 'C', 'DandC']}
        )
        mdl.setup_model()
        mdl.solve_model()
        sa_output = pd.concat([sa_output, pd.DataFrame.from_dict({
            'SA Param': np.array([par]),
            **{k: np.array([v]) for k, v in mdl_params.items()},
            'Lower bound': np.array([mdl.results['Problem'][0]['Lower bound']]),
            'Upper bound': np.array([mdl.results['Problem'][0]['Upper bound']]),
            'Optimality Gap': np.array([mdl.results['Solution'][0]['Gap']]),
            'Status': np.array([mdl.results['Solution'][0]['Status']]),
            'Objective': np.array([mdl.results['Solution'][0]['Objective']['objective']['Value']])
        })], ignore_index=True)
        print(f"Ran SA for SA param {par} = {val}")

sa_output.to_csv('model_runs/laes/laes.csv', index=False)
