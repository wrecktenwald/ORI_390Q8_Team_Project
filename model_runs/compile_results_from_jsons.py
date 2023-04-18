"""
Compile results from model JSON files in search paths
"""


import sys
import numpy as np
import pandas as pd

sys.path.append('../ORI_390Q8_Team_Project')

from utilities import read_model_results_json


search_paths = [
    'model_runs/daily_med_to_long_dur/',
    'model_runs/weekly_long_dur/',
]

# TODO: check for all JSON files in search_paths
# TODO: attempt to read contents with read_model_results_json
