"""
Sensitivity analysis for long term duration storage modeleled at a weekly level
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
    sa_output = pd.read_csv('model_runs/weekly_long_dur/weekly_long_dur_output.csv')  # update during runs 
except FileNotFoundError:
    print('Warning will create .csv')
    sa_output = pd.DataFrame()  # fill during runs

wks = 52  # weekly, run one year in circular model
hrs_p = {
    pt: dam_avg.loc[dam_avg['SETTLEMENT_POINT'] == pt].sort_values(['MONTH', 'DAY', 'HOUR_ENDING'])['PRICE'].to_numpy()
    for pt in [
        'HB_BUSAVG', 
        'HB_HOUSTON', 
        'HB_HUBAVG', 
        'HB_NORTH', 
        'HB_PAN',
        'HB_SOUTH', 
        'HB_WEST', 
        'LZ_AEN', 
        'LZ_CPS', 
        'LZ_HOUSTON', 
        'LZ_LCRA',
        'LZ_NORTH', 
        'LZ_RAYBN', 
        'LZ_SOUTH', 
        'LZ_WEST'
    ]
}
wks_p_ids = {_k_: np.arange(len(_v_)) // (7 * 24) for _k_, _v_ in hrs_p.items()}
wks_p = {_k_: np.bincount(wks_p_ids[_k_], hrs_p[_k_]) / np.bincount(wks_p_ids[_k_]) for _k_ in hrs_p.keys()}

big_S = 1  # normalize to 1 MWh
baseline = dict(
    R=0.95,  # estimate fuel type costs
    batch_d_rt=21,  # attempt to replace natural gas utility type storage with similar seasonality
    batch_c_rt=31,  # attempt to replace natural gas utility type storage with similar seasonality
    vd=0.0,  # estimate costs?
    vc=0.0,  # estimate costs? 
    p=wks_p['LZ_AEN'][:wks]
)
sa_params = dict(
    R=np.arange(0, 1 + 0.05, 0.01),
    batch_d_rt=np.arange(40, 0, -1),
    batch_c_rt=np.arange(40, 0, -1),
    vd=np.concatenate((np.arange(0, 10 + 0.25, 0.25), np.arange(11, 20 + 1, 1), np.arange(25, 50 + 5, 5), np.arange(60, 100 + 10, 10))),
    vc=np.concatenate((np.arange(0, 10 + 0.25, 0.25), np.arange(11, 20 + 1, 1), np.arange(25, 50 + 5, 5), np.arange(60, 100 + 10, 10))),
    p=wks_p
)

mdl_params = baseline.copy()
mdl = vp_v4_0(
    valid_pair_span=wks,
    periods=wks,
    S=np.full(wks, big_S),  # no system degradation
    r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
    solver='glpk',
    D=np.full(wks, big_S / mdl_params['batch_d_rt']),
    C=np.full(wks, big_S / mdl_params['batch_c_rt']),
    **{k: v for k, v in mdl_params.items()}
)
mdl.setup_model()
mdl.solve_model()
sa_output = pd.concat([sa_output, pd.DataFrame.from_dict({
    'SA Param': np.array(['Baseline']),
    **{k: np.array([np.mean(v)]) if isinstance(v, type(np.array([]))) else np.array([v]) for k, v in mdl_params.items()},
    'Lower bound': np.array([mdl.results['Problem'][0]['Lower bound']]),
    'Upper bound': np.array([mdl.results['Problem'][0]['Upper bound']]),
    'Optimality Gap': np.array([mdl.results['Solution'][0]['Gap']]),
    'Status': np.array([mdl.results['Solution'][0]['Status']]),
    'Objective': np.array([mdl.results['Solution'][0]['Objective']['objective']['Value']])
})], ignore_index=True)
print('Ran SA for baseline')
mdl.write_to_json(filename='model_runs/weekly_long_dur/baseline.json')

for par, vals in sa_params.items():  # iterate over models editing baseline by param
    if isinstance(vals, dict):
        for par_name, val in vals.items():
            mdl_params = baseline.copy()
            mdl_params.update({par: val})
            mdl = vp_v4_0(
                valid_pair_span=wks,
                periods=wks,
                S=np.full(wks, big_S),  # no system degradation
                r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
                solver='glpk',
                D=np.full(wks, big_S / mdl_params['batch_d_rt']),
                C=np.full(wks, big_S / mdl_params['batch_c_rt']),
                **{k: v for k, v in mdl_params.items()}
            )
            mdl.setup_model()
            mdl.solve_model()
            sa_output = pd.concat([sa_output, pd.DataFrame.from_dict({
                'SA Param': np.array([par + ': ' + par_name + ' (Mean Ref.)']),
                **{k: np.array([np.mean(v)]) if isinstance(v, type(np.array([]))) else np.array([v]) for k, v in mdl_params.items()},
                'Lower bound': np.array([mdl.results['Problem'][0]['Lower bound']]),
                'Upper bound': np.array([mdl.results['Problem'][0]['Upper bound']]),
                'Optimality Gap': np.array([mdl.results['Solution'][0]['Gap']]),
                'Status': np.array([mdl.results['Solution'][0]['Status']]),
                'Objective': np.array([mdl.results['Solution'][0]['Objective']['objective']['Value']])
            })], ignore_index=True)
            print(f"Ran SA for SA param {par + ': ' + par_name} = {val}")
    else:
        for val in vals:
            mdl_params = baseline.copy()
            mdl_params.update({par: val})
            mdl = vp_v4_0(
                valid_pair_span=wks,
                periods=wks,
                S=np.full(wks, big_S),  # no system degradation
                r=0.05,  # 5 to 10% noted to be a good assumption by Dr. Leibowicz
                solver='glpk',
                D=np.full(wks, big_S / mdl_params['batch_d_rt']),
                C=np.full(wks, big_S / mdl_params['batch_c_rt']),
                **{k: v for k, v in mdl_params.items()}
            )
            mdl.setup_model()
            mdl.solve_model()
            sa_output = pd.concat([sa_output, pd.DataFrame.from_dict({
                'SA Param': np.array([par]),
                **{k: np.array([np.mean(v)]) if isinstance(v, type(np.array([]))) else np.array([v]) for k, v in mdl_params.items()},
                'Lower bound': np.array([mdl.results['Problem'][0]['Lower bound']]),
                'Upper bound': np.array([mdl.results['Problem'][0]['Upper bound']]),
                'Optimality Gap': np.array([mdl.results['Solution'][0]['Gap']]),
                'Status': np.array([mdl.results['Solution'][0]['Status']]),
                'Objective': np.array([mdl.results['Solution'][0]['Objective']['objective']['Value']])
            })], ignore_index=True)
            print(f"Ran SA for SA param {par} = {val}")

sa_output.to_csv('model_runs/weekly_long_dur/weekly_long_dur_output.csv', index=False)
