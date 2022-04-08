from enum import Enum
from copy import copy
from datetime import timedelta
from itertools import groupby, repeat
import ocel_features.util.data_organization as do


def timeframe_sublog(log, params):
    start_time = params['start_time']
    time_delta = params['timedelta']
    end_time = start_time + time_delta
    log_events = []

    for e, v in log['ocel:events'].items():
        if start_time <= v['ocel:timestamp'] <= end_time:
            log_events.append(e)
        elif log_events:
            # if it is done iterating
            return log_events

    return log_events


# split up by day
def time_day_decomp(log, params):
    events = log['ocel:events']
    return [list(g) for k, g in
            groupby([e for e, v in events.items()],
                    key=lambda d: events[d]['ocel:timestamp'].date())]


# split based on some timedelta
def time_delta_decomp(log, params):
    event_bins = []
    curr_time = params['start_time'] + params['timedelta']
    events = log['ocel:events']
    curr_bin = []

    for k, v in events.items():
        e_time = v['ocel:timestamp']
        if params['start_time'] <= e_time <= curr_time:
            curr_bin.append(k)
        else:
            event_bins.append(curr_bin)
            curr_bin = []
            curr_time = curr_time + params['timedelta']

    if curr_bin:
        event_bins.append(curr_bin)

    return event_bins


# split based on week and specific weekdays
def time_week_days_decomp(log, params):
    event_bins = []
    events = log['ocel:events']
    e_time = params['start_time']
    curr_week = (e_time - timedelta(days=e_time.weekday())).date()
    curr_bin = []

    for e, v in events.items():
        e_time = v['ocel:timestamp'].date()
        if (curr_week + timedelta(weeks=1)) <= e_time:
            event_bins.append(curr_bin)
            curr_week = (e_time - timedelta(days=e_time.weekday()))
            curr_bin = []

        # only accept wanted days of week
        if e_time.weekday() in params['included_days']:
            curr_bin.append(e)

    if curr_bin:
        event_bins.append(curr_bin)

    return event_bins


def time_single_week_days(log, params):
    events = log['ocel:events']
    start = params['start_time']
    end = params['end_time']
    days = params['included_days']
    dates = [(start + timedelta(days=x)).date()
             for x in range((end-start).days + 1)
             if (start+timedelta(days=x)).weekday() in days]
    # create lookup
    dates = {date: i for i, date in enumerate(dates)}
    event_bins = [[] for i in repeat(None, len(dates))]

    for e, v in events.items():
        e_time = v['ocel:timestamp']
        e_date = e_time.date()
        # only accept wanted days of week
        if e_time.weekday() in days and e_date in dates:
            event_bins[dates[e_date]].append(e)

    return event_bins


class Decompositions(Enum):
    TIME_DELTA = (time_delta_decomp, ['start_time', 'timedelta'])
    TIME_DAY = (time_day_decomp, [])
    TIME_SINGLE_WEEK_DAYS = (time_single_week_days, ['included_days'])
    TIME_WEEK_DAYS = (time_week_days_decomp, ['included_days', 'start_time',
                                              'end_time'])
    TIMEFRAME_SUBLOG = (timeframe_sublog, ['start_time', 'timedelta'])


def decompose_log(log, decomp=Decompositions.TIME_DELTA, params=None):
    if params is None:
        # set to default decomp and params (always works)
        decomp = Decompositions.TIME_DELTA
        params = {}
        params['timedelta'] = timedelta(days=1)

    if 'start_time' not in params:
        first_e = list(log['ocel:events'].keys())[0]
        params['start_time'] = log['ocel:events'][first_e]['ocel:timestamp']

    if 'end_time' not in params:
        last_e = list(log['ocel:events'].keys())[-1]
        params['end_time'] = log['ocel:events'][last_e]['ocel:timestamp']
    if not do.check_params(params, decomp.value[1]):
        return

    sublogs = []
    event_bins = decomp.value[0](log, params)
    # create the logs
    for eb in event_bins:
        new_log = copy(log)
        new_log['ocel:events'] = {k: log['ocel:events'][k] for k in eb}
        sublogs.append(new_log)

    return sublogs

