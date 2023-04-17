'''
Average RTM settlement price variation 
based on input frequency: HOUR, DAY, MONTH (data from 2019, 2020, 2022)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ercot_prices import ercot_dam_prices, ercot_rtm_prices

df = pd.read_csv('data/ERCOT/post_processing/rtm_avg.csv')

# All settlement points for reference (HB = Hub, LZ = Load Zone)
# 'HB_BUSAVG', 'HB_HOUSTON', 'HB_HUBAVG', 'HB_NORTH', 'HB_PAN',
# 'HB_SOUTH', 'HB_WEST', 'LZ_AEN', 'LZ_CPS', 'LZ_HOUSTON', 'LZ_LCRA',
# 'LZ_NORTH', 'LZ_RAYBN', 'LZ_SOUTH', 'LZ_WEST'

def createPriceGraph(frequency, settlement):
    plt.rcParams["figure.figsize"] = (10,6)
    data = df[df['SETTLEMENT_POINT'] == settlement].groupby(by=[frequency], as_index=False)['PRICE'].mean()
    f, ax = plt.subplots()
    ax.plot(data[frequency], data['PRICE'], 'r')
    ax.set_xlabel(frequency)
    ax.set_ylabel('PRICE')
    ax.set_xticks(data[frequency])
    if frequency == 'DAY':
        ax.set_title('DAILY average RTM price variation')
    else:
        ax.set_title(frequency + 'LY average RTM price variation')

createPriceGraph('HOUR', 'LZ_LCRA')
createPriceGraph('DAY', 'LZ_LCRA')
createPriceGraph('MONTH', 'LZ_LCRA')