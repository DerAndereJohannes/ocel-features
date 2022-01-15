import pandas as pd
from statistics import mean, stdev, median, mode


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


def equal_width_bins():
    pass


def equal_frequency_bins():
    pass


def equal_time_bins():
    pass
