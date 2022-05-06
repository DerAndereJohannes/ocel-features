import pandas as pd
from statistics import mean, stdev, median, mode
from enum import Enum


_ADDED_STATS = [('mean', mean), ('stdev', stdev),
                ('median', median), ('mode', mode)]


def conv_dict_to_pandas(feature_list, obj_dict):
    row_object = list()
    data_x = [None]*len(obj_dict.keys())
    for i, row in enumerate(obj_dict.keys()):
        row_object.append(row)
        data_x[i] = obj_dict[row]

    df = pd.DataFrame(data_x)
    df.transpose()
    df.columns = feature_list

    return df, row_object


def get_basic_stats(col_name_pre, series):
    names_added = []
    values_added = []

    if series:
        names_added = [f'{col_name_pre}_{stat[0]}' for stat in _ADDED_STATS]
        values_added = [stat[1](series) for stat in _ADDED_STATS]

    return names_added, values_added


def equal_frequency_bins(series, freq):
    return [int(i/freq) for i in len(series)]


def time_bins(series, timediff):
    return [int((ts - series[0]) / timediff) for ts in series]


def equal_time_width_bins(series):
    time_width = (series[-1] - series[0]) / len(series)
    return time_bins(series, time_width)


def flatten_list(t):
    return [item for sublist in t for item in sublist]


def get_min(series):
    return min(series)


def get_max(series):
    return max(series)


def get_avg(series):
    return mean(series)


def get_count(series):
    return len(series)


def get_sum(series):
    return sum(series)


class Operators(Enum):
    COUNT = (get_count,)
    SUM = (get_sum,)
    AVG = (get_avg,)
    MAX = (get_max,)
    MIN = (get_min,)


def execute_operator(op, series):
    return op.value[0](series)


def operator_name(op):
    return Operators(op).name


def check_params(params, req_params):
    # check param properties
    if req_params and not params:
        print('Please provide parameters',
              f'{req_params} to use this variant.')
        return 0
    elif params:
        not_contained = []
        for p in req_params:
            if p not in params:
                not_contained.append(p)
        if not_contained:
            print('Please additionally provide parameters',
                  f'{req_params} to use this variant.')
            return 0

    return 1
