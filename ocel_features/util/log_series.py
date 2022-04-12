import ocel_features.util.data_organization as do
import ocel_features.util.ocel_helper as oh
import ocel_features.util.experimental.convert_neo4j as n4j
import itertools
from enum import Enum


# binning helper functions
def remove_bin_gaps(bin_list):
    id_dict = {b: i for i, b in enumerate(set(bin_list))}
    return [id_dict[b] for b in bin_list]


# binning functions ###
def get_equal_time_binning(log):
    vol_series = [v['ocel:timestamp'] for v in log['ocel:events'].values()]
    return do.equal_time_width_bins(vol_series)


# binning based on timedelta
def get_time_diff_binning(log, timediff):
    vol_series = [v['ocel:timestamp'] for v in log['ocel:events'].values()]
    return do.time_bins(vol_series, timediff)


# binning based on day of the week and whole week
def get_date_range_binning(log, days):
    bins = n4j.split_by_days_of_week(log, days)
    date_bins = []
    for i, b in enumerate(bins):
        date_bins.extend([i for _ in range(len(bins[b]['ocel:events']))])

    return date_bins


def activity_count(series, log, params):
    return len(series)


def total_obj_count(series, log, params):
    return sum([len(series[e]['ocel:omap']) for e in series])


def total_unique_obj_count(series, log, params):
    return len({len(series[e]['ocel:omap']) for e in series})


def object_property_operator(series, log, params):
    if 'o_types' not in params:
        o_types = log['ocel:global-log']['ocel:object-types']
    else:
        o_types = params['o_types']

    prop = params['property']
    operator = params['operator'].value[0]

    unique_obj = set()
    result_series = []
    objects = log['ocel:objects']

    for ev in series.values():
        for o in ev['ocel:omap']:
            if objects[o]['ocel:type'] in o_types \
               and prop in objects[o]['ocel:ovmap']\
               and o not in unique_obj:

                result_series.append(objects[o]['ocel:ovmap'][prop])
                unique_obj.add(o)

    return operator(result_series)


def event_property_operator(series, log, params):
    if 'activities' not in params:
        activities = oh.get_activity_names(log)
    else:
        activities = params['activities']

    prop = params['property']
    operator = params['operator'].value[0]

    result_series = []

    for k, v in series.items():
        if v['ocel:activity'] in activities and prop in v['ocel:vmap']:
            result_series.append(v['ocel:vmap'][prop])

    return operator(result_series)


class LogFunctions(Enum):
    ACTIVITY_COUNT = (activity_count, [])
    TOTAL_OBJ_COUNT = (total_obj_count, [])
    UNIQUE_OBJ_COUNT = (total_unique_obj_count, [])
    OBJ_PROPERTY_OPERATOR = (object_property_operator, ['operator',
                                                        'property'])
    EV_PROPERTY_OPERATOR = (event_property_operator, ['operator', 'property'])


# main function
def apply_func_to_binning(log, bin_list, func=LogFunctions.ACTIVITY_COUNT,
                          params=None):

    if not do.check_params(params, func.value[1]):
        return

    curr_id = bin_list[0]
    bin_volatility = [0 for i in range(bin_list[-1]+1)]
    head = 0
    tail = 0

    for i, b in enumerate(bin_list):
        if b != curr_id:
            # get series of events
            series = dict(itertools.islice(log['ocel:events'].items(),
                                           head, tail))
            bin_volatility[curr_id] = func.value[0](series, log, params)
            curr_id = b
            head = tail

        tail = tail + 1

    return bin_volatility


# manipulate the series (post-processing) ####
def series_differences_absolute(series):
    out = [0]*(len(series)-1)
    for i in (range(len(series) - 1)):
        out[i] = series[i+1] - series[i]

    return out


def series_differences_percentage(series, norm=False):
    out = [0]*(len(series)-1)
    for i in (range(len(series) - 1)):
        if series[i] != 0:
            out[i] = (series[i+1] - series[i]) / series[i]
        else:
            out[i] = 1

    if norm:
        max_val = max([abs(o) for o in out])
        return list(map(lambda x: x/max_val, out))

    return out
