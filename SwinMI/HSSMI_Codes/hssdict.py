import numpy as np
from typing import List, Tuple, Dict

def format_channel_location_dict(channel_list, location_list):
    location_list = np.array(location_list)
    output = {}
    for channel in channel_list:
        if len(np.argwhere(location_list == channel)):
            location = (np.argwhere(location_list == channel)[0]).tolist()
            output[channel] = location
        else:
            output[channel] = None
    return output

HSS_CHANNEL_LIST = [
    'FC3', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CP4'
]

HSS_LOCATION_LIST = [
    ['-',   '-',   '-',   '-',   '-',    '-',   '-'],
    ['-',   '-',   '-',   '-',   '-',    '-',   '-'],
    ['-',   'FC3', '-',   '-',   '-',    'FC4', '-'],
    ['C5',  'C3',  'C1',  'CZ',  'C2',   'C4',  'C6'],
    ['-',   'CP3', '-',   '-',   '-',    'CP4', '-'],
    ['-',   '-',   '-',   '-',   '-',    '-',   '-'],
    ['-',   '-',   '-',   '-',   '-',    '-',   '-']
]

HSS_LOCATION_DICT = format_channel_location_dict(HSS_CHANNEL_LIST,HSS_LOCATION_LIST)