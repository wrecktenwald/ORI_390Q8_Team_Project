"""
Average ERCOT Day Ahead Market (DAM), Real Time Market (RTM) Load Zone (LZ) and Hub prices for 2019-2022
"""

import numpy as np
import pandas as pd

from ercot_prices import ercot_dam_prices, ercot_rtm_prices


years = [2019, 2020, 2022]  # exclude Uri year for less anomalous price profile

dam_by_yr = {yr: ercot_dam_prices(year=yr) for yr in years}
rtm_by_yr = {yr: ercot_rtm_prices(year=yr) for yr in years}

dam = pd.concat(dam_by_yr.values())
rtm = pd.concat(rtm_by_yr.values())

dam_avg = dam.drop(
    labels=['YEAR', 'REPEATED_HOUR_FLAG'], 
    axis=1).groupby(
    by=['MONTH', 'DAY', 'HOUR_ENDING', 'SETTLEMENT_POINT'], 
    as_index=False
).mean()
rtm_avg = rtm.drop(
    labels=['YEAR', 'INTERVAL','REPEATED_HOUR_FLAG'], 
    axis=1).groupby(
    by=['MONTH', 'DAY', 'HOUR', 'SETTLEMENT_POINT', 'TYPE'], 
    as_index=False
).mean()

# Remove Febraury 29th as there is not sufficient data to average this
dam_avg = dam_avg.loc[~((dam_avg['MONTH'] == 2) & (dam_avg['DAY'] == 29))]
dam_avg.reset_index(drop=True, inplace=True)
rtm_avg = rtm_avg.loc[~((rtm_avg['MONTH'] == 2) & (rtm_avg['DAY'] == 29))]
rtm_avg.reset_index(drop=True, inplace=True)

# Perform final check validating there are 8760 rows per settlement point
dam_pvt_ck = pd.pivot_table(dam_avg, values='PRICE', index=['MONTH', 'DAY', 'HOUR_ENDING'], columns='SETTLEMENT_POINT')
rtm_pvt_ck = pd.pivot_table(dam_avg, values='PRICE', index=['MONTH', 'DAY', 'HOUR_ENDING'], columns='SETTLEMENT_POINT')
if dam_pvt_ck.shape[0] != 8760:
    raise ValueError('ERCOT Day Ahead Market (DAM) average data does not contain 8760 hours')
if rtm_pvt_ck.shape[0] != 8760:
    raise ValueError('ERCOT Real Time Market (RTM) average data does not contain 8760 hours')

ercot_data_folder = f"{__file__.split('ORI_390Q8_Team_Project')[0]}/ORI_390Q8_Team_Project/data/ERCOT/"
dam_avg.to_csv(f"{ercot_data_folder}/post_processing/dam_avg.csv", index=False)
rtm_avg.to_csv(f"{ercot_data_folder}/post_processing/rtm_avg.csv", index=False)
