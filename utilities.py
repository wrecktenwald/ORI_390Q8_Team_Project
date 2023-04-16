"""
Utilities
"""

import time


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
