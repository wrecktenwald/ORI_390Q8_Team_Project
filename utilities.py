"""
Utilities
"""

import time
import json
import pandas as pd


def elapsed_time_display(
    start,
    insert_phrase=None
):
    """
    Display elapsed time

    Parameters
    __________
    start : time.time() object
        Start time
    insert_phrase : str, default None
        Phrase to insert into print out
    """
    
    elapsed = time.time() - start
    sec = elapsed % (24 * 3600)
    hour = elapsed // 3600
    elapsed %= 3600
    min = elapsed // 60
    elapsed %= 60

    print(f"Elapsed time{insert_phrase}: %02d:%02d:%02d" % (hour, min, sec)) 


def read_model_results_json(
    filename
):
    """
    Read critical results data in JSON written to JSON from a pyomo model 

    Parameters
    __________
    filename : str
        Filename and location to read JSON, must end in '.json'
    """

    f = open(filename)
    _json_ = json.load(f)
    f.close()

    if len(_json_['Solution']) > 2:
        raise ValueError("Unexpected length of 'Solution' (greater than 2)")

    results =  {
        'Lower bound': _json_['Problem'][0]['Lower bound'],
        'Upper bound': _json_['Problem'][0]['Upper bound'],
        'Optimality Gap': _json_['Solution'][1]['Gap'],
    }

    try:
        results['Objective'] = _json_['Solution'][1]['Objective']['objective']['Value']
    except KeyError:
        try:
            results['Objective'] = _json_['Solution'][1]['Objective']['OBJ']['Value']
        except KeyError:
            print(f"Check returned 'Objective' for {filename}")
            results['Objective'] = _json_['Solution'][1]['Objective']

    results.update({
        'Status': _json_['Solution'][1]['Status'], 
        'Decision Variables': pd.Series({k: v['Value'] for k, v in _json_['Solution'][1]['Variable'].items()})
    })

    return results
