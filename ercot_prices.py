"""
Load ERCOT Day Ahead Market (DAM), Real Time Market (RTM) Load Zone (LZ) and Hub prices
"""

import numpy as np
import pandas as pd
from datetime import datetime


def ercot_dam_prices(
        year
):
    """
    Load ERCOT DAM LZ and Hub prices

    Parameters
    __________
    year : int
        Year to load prices for, 2019-2023

    Returns
    _______
    DataFrame
    """

    raw_data_by_month = pd.read_excel(
        io=f"{__file__.split('ORI_390Q8_Team_Project')[0]}/ORI_390Q8_Team_Project/data/ERCOT/Historical_DAM_Load_Zone_and_Hub_Prices/{year}.xlsx",
        sheet_name=None  # loads all sheets
    )

    raw_data = pd.concat(raw_data_by_month.values(), ignore_index=True)
    raw_data['Date'] = pd.to_datetime(raw_data['Delivery Date'])
    raw_data['YEAR'] = [_.year for _ in raw_data['Date']]
    raw_data['MONTH'] = [_.month for _ in raw_data['Date']]
    raw_data['DAY'] = [_.day for _ in raw_data['Date']]
    raw_data['HOUR_ENDING'] = [int(_[:2]) for _ in raw_data['Hour Ending']]
    raw_data.rename(
        {
            'Repeated Hour Flag': 'REPEATED_HOUR_FLAG',
            'Settlement Point': 'SETTLEMENT_POINT',
            'Settlement Point Price': 'PRICE'
        }, 
        axis=1,
        inplace=True
    )

    data = raw_data[['YEAR', 'MONTH', 'DAY', 'HOUR_ENDING', 'REPEATED_HOUR_FLAG', 'SETTLEMENT_POINT', 'PRICE']]

    return data



def ercot_rtm_prices(
        year
):
    """
    Load ERCOT RTM LZ and Hub prices

    Parameters
    __________
    year : int
        Year to load prices for, 2019-2023

    Returns
    _______
    DataFrame
    """

    raw_data_by_month = pd.read_excel(
        io=f"{__file__.split('ORI_390Q8_Team_Project')[0]}/ORI_390Q8_Team_Project/data/ERCOT/Historical_RTM_Load_Zone_and_Hub_Prices/{year}.xlsx",
        sheet_name=None  # loads all sheets
    )

    raw_data = pd.concat(raw_data_by_month.values(), ignore_index=True)
    raw_data['Date'] = pd.to_datetime(raw_data['Delivery Date'])
    raw_data['YEAR'] = [_.year for _ in raw_data['Date']]
    raw_data['MONTH'] = [_.month for _ in raw_data['Date']]
    raw_data['DAY'] = [_.day for _ in raw_data['Date']]
    raw_data['HOUR'] = [int(_) for _ in raw_data['Delivery Hour']]
    raw_data['INTERVAL'] = [int(_) for _ in raw_data['Delivery Interval']]
    raw_data.rename(
        {
            'Repeated Hour Flag': 'REPEATED_HOUR_FLAG',
            'Settlement Point Name': 'SETTLEMENT_POINT',
            'Settlement Point Type': 'TYPE',
            'Settlement Point Price': 'PRICE'
        }, 
        axis=1,
        inplace=True
    )

    data = raw_data[['YEAR', 'MONTH', 'DAY', 'HOUR', 'INTERVAL', 'REPEATED_HOUR_FLAG', 'SETTLEMENT_POINT', 'TYPE', 'PRICE']]

    return data
