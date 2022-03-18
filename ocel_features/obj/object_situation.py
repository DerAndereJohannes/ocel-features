import pandas as pd
import numpy as np
import networkx as nx
import math
import pytz
from datetime import datetime
from collections import Counter
from itertools import combinations
from enum import Enum
from copy import copy
from ocel_features.util.data_organization import Operators
import ocel_features.obj.object_point as op
import ocel_features.obj.object_global as og
import ocel_features.util.ocel_helper as oh


class Situation:
    def __init__(self, log, graph, params):
        self._log = log
        self._sublog = params['sublog']
        self._graph = graph
        self._objects = self._sublog['ocel:objects']
        self._events = self._sublog['ocel:events']
        self._situation_event = params['situation_event']

        min_time_key = min(self._events,
                           key=lambda k: self._events[k]['ocel:timestamp'])
        max_time_key = max(self._events,
                           key=lambda k: self._events[k]['ocel:timestamp'])
        self._mintime = log['ocel:events'][min_time_key]['ocel:timestamp']
        self._maxtime = log['ocel:events'][max_time_key]['ocel:timestamp']

        self._df = pd.DataFrame()
        self._targetdf = params['target_feature']

    def __repr__(self):
        return f'Situation({self._targetdf.to_dict()})'

    def get_latest_an_properties(self, an):
        events = self._log['ocel:events']
        recent_e = None
        for e in self._events:
            if events[e]['ocel:activity'] == an:
                if not recent_e or (recent_e and
                                    (events[e]['ocel:timestamp'] >
                                        events[recent_e]['ocel:timestamp'])):
                    recent_e = e

        for k, v in events[recent_e]['ocel:vmap'].items():
            key = f'att:event:latest:{k}'
            self._df[key] = [v]

    def get_op_an_properties(self, an: str, op: Operators):
        events = self._log['ocel:events']
        op_function = op.value[0]
        op_name = op.name
        values = {}
        for e in self._events:
            if events[e]['ocel:activity'] == an:
                for k, v in events[e]['ocel:vmap'].items():
                    if isinstance(v, (int, float)):
                        if k not in values:
                            values[k] = [v]
                        else:
                            values[k].append(v)

        for k, v in values.items():
            key = f'att:{an}:{k}:{op_name}:'
            self._df[key] = [op_function(v)]

    def get_latest_ot_properties(self, ot: str):
        ot = ot.pop() if isinstance(ot, set) else ot
        objects = self._log['ocel:objects']
        events = self._log['ocel:events']
        recent_o = None
        recent_e = None
        for o in self._objects:
            if objects[o]['ocel:type'] == ot:
                e = self._graph.nodes[o]['object_events'][0]
                if not recent_o or (recent_o and
                                    (events[e]['ocel:timestamp'] >
                                        events[recent_e]['ocel:timestamp'])):
                    recent_o = o
                    recent_e = e

        for k, v in objects[recent_o]['ocel:ovmap'].items():
            key = f'att:{ot}:latest:{k}'
            self._df[key] = [v]

    def get_op_ot_properties(self, ot: str, op: Operators):
        ot = ot.pop() if isinstance(ot, set) else ot
        objects = self._log['ocel:objects']
        op_function = op.value[0]
        op_name = op.name
        values = {}
        for o in self._objects:
            if objects[o]['ocel:type'] == ot:
                for k, v in objects[o]['ocel:ovmap'].items():
                    if isinstance(v, (int, float)):
                        if k not in values:
                            values[k] = [v]
                        else:
                            values[k].append(v)

        for k, v in values.items():
            key = f'att:{ot}:{k}:{op_name}:'
            self._df[key] = [op_function(v)]

    def get_obj_attributes(self):
        for o in self._objects:
            for k, v in self._log['ocel:objects'][o]['ocel:ovmap'].items():
                ot = self._log['ocel:objects'][o]['ocel:type']
                key = f'att:{ot}:{o}:{k}'
                self._df[key] = [v]

    def get_ot_attributes(self, ot: set = None):
        if not isinstance(ot, set):
            ot = set(self._log['ocel:global-log']['ocel:object-types'])
        for o in self._objects:
            o_ot = self._log['ocel:objects'][o]['ocel:type']
            if o_ot in ot:
                for k, v in self._log['ocel:objects'][o]['ocel:ovmap'].items():
                    key = f'att:{o_ot}:{o}:{k}'
                    self._df[key] = [v]

    def get_agg_ot_attributes(self, ot: set = None):
        # only goes over values
        if not isinstance(ot, set):
            ot = set(self._log['ocel:global-log']['ocel:object-types'])
        curr_col = copy(self._df.columns)
        for o in self._objects:
            o_ot = self._log['ocel:objects'][o]['ocel:type']
            if o_ot in ot:
                for k, v in self._log['ocel:objects'][o]['ocel:ovmap'].items():
                    ot = self._log['ocel:objects'][o]['ocel:type']
                    key = f'agg:{ot}:{k}'
                    if key in curr_col:
                        continue  # if columns already exists
                    if isinstance(v, (int, float)):
                        if key in self._df:
                            self._df[key] += v
                        else:
                            self._df[key] = [v]

    def get_event_attributes(self):
        for e in self._events:
            for k, v in self._log['ocel:events'][e]['ocel:vmap'].items():
                an = self._log['ocel:events'][e]['ocel:activity']
                key = f'att:{an}:{e}:{k}'
                self._df[key] = [v]

    def get_agg_activity_attributes(self):
        # only goes over values
        curr_col = copy(self._df.columns)
        for e in self._events:
            an = self._log['ocel:events'][e]['ocel:activity']
            for k, v in self._log['ocel:events'][e]['ocel:vmap'].items():
                key = f'agg:{an}:{k}'
                if key in curr_col:
                    continue  # if columns already exists
                if isinstance(v, (int, float)):
                    if key in self._df:
                        self._df[key] += v
                    else:
                        self._df[key] = [v]

    def get_object_point(self, fname, finput, oids):

        point = op.Object_Based(self._log, self._graph, oids)
        func = getattr(point, fname)

        func(*finput)

        df_dict = point.df_full().to_dict(orient='records')
        for row in df_dict:
            roid = row['oid']
            for k, v in row.items():
                if op._FEATURE_PREFIX in k:
                    self._df[f'{roid}:{k}'] = [v]

    def get_object_global(self, fname, finput, oids):

        point = og.Object_Global(self._log, self._graph)
        func = getattr(point, fname)

        func(*finput)

        df_dict = point.df_full().to_dict(orient='records')[0]

        for k, v in df_dict.items():
            if og._FEATURE_PREFIX in k:
                self._df[k] = [v]

    def df_numeric(self):
        return self._df.select_dtypes(include=np.number)

    def __add__(self, other):
        if type(self) == type(other):
            if all(self._df.columns == other._df.columns):
                return pd.concat([self._df, other._df],
                                 ignore_index=True, sort=True)


def validate_event_choice_target(log, graph, target_event, params=None):
    an = params['activities']
    return log['ocel:events'][target_event]['ocel:activity'] in an


def event_choice_target(log, graph, target_event, params=None):
    ed = log['ocel:events'][target_event]

    sobjects = ed['ocel:omap']
    sevents = oh.get_relevant_events(log, graph, sobjects,
                                     oh.get_last_event(log))
    situation_sublog = oh.create_sublog(log, sobjects, sevents)
    target_feature = pd.DataFrame([{'event choice': ed['ocel:activity']}])
    return {'target_feature': target_feature,
            'situation_features': None,
            'sublog': situation_sublog,
            'situation_event': target_event
            }


def validate_event_property_target(log, graph, target_event, params=None):
    prop = params['property']
    return prop in log['ocel:events'][target_event]['ocel:vmap']


def event_property_target(log, graph, target_event, params=None):
    prop = params['property']
    ed = log['ocel:events'][target_event]

    sobjects = ed['ocel:omap']
    sevents = oh.get_relevant_events(log, graph, sobjects, target_event)
    situation_sublog = oh.create_sublog(log, sobjects, sevents)

    target_feature = pd.DataFrame([{f'event <{prop}>': ed['ocel:vmap'][prop]}])

    return {'target_feature': target_feature,
            'situation_features': None,
            'sublog': situation_sublog,
            'situation_event': target_event
            }


def validate_event_wait(log, graph, target_event, params=None):
    ev = log['ocel:events']
    ed = ev[target_event]

    if not ed['ocel:omap']:
        return False

    for o in ed['ocel:omap']:
        oe = graph.nodes[o]['object_events']
        if target_event != oe[0]:
            return True
    return False


def event_wait_target(log, graph, target_event, params=None):
    ev = log['ocel:events']
    ed = ev[target_event]

    oldest_time = datetime.max.replace(tzinfo=pytz.utc)
    for o in ed['ocel:omap']:
        oe = graph.nodes[o]['object_events']
        if target_event != oe[0]:
            i = oe.index(target_event) - 1
            if i >= 0:
                e_time = ev[oe[i]]['ocel:timestamp']
                if e_time < oldest_time:
                    oldest_time = e_time

    sobjects = ed['ocel:omap']
    sevents = oh.get_relevant_events(log, graph, sobjects, target_event)
    situation_sublog = oh.create_sublog(log, sobjects, sevents)

    target_feature = pd.DataFrame([{'wait until event':
                                    ed['ocel:timestamp'] - oldest_time}])

    return {'target_feature': target_feature,
            'situation_feature': None,
            'sublog': situation_sublog,
            'situation_event': target_event
            }


def validate_event_duration(log, graph, target_event, params=None):
    ev = log['ocel:events']
    ed = ev[target_event]

    if not ed['ocel:omap']:
        return False

    for o in ed['ocel:omap']:
        oe = graph.nodes[o]['object_events']
        if target_event != oe[-1]:
            return True
    return False


def event_duration_target(log, graph, target_event, params=None):
    ev = log['ocel:events']
    ed = ev[target_event]

    youngest_time = datetime.max.replace(tzinfo=pytz.utc)
    last_event = None
    for o in ed['ocel:omap']:
        oe = graph.nodes[o]['object_events']
        if target_event != oe[-1]:
            i = oe.index(target_event) + 1
            if i < len(oe):
                e_time = ev[oe[i]]['ocel:timestamp']
                if e_time < youngest_time:
                    youngest_time = e_time
                    last_event = oe[i]

    sobjects = ed['ocel:omap']
    sevents = oh.get_relevant_events(log, graph, sobjects, last_event)
    situation_sublog = oh.create_sublog(log, sobjects, sevents)

    target_feature = pd.DataFrame([{'wait after event':
                                    youngest_time - ed['ocel:timestamp']}])

    return {'target_feature': target_feature,
            'situation_feature': None,
            'sublog': situation_sublog,
            'situation_event': target_event
            }


def validate_object_choice(log, graph, target_event, params=None):
    an = params['activities']
    ot = params['object_type']
    events = log['ocel:events'][target_event]
    objects = log['ocel:objects']

    return events['ocel:activity'] in an \
        and [o for o in events['ocel:omap'] if objects[o]['ocel:type'] == ot]


def object_choice_target(log, graph, target_event, params=None):
    ot = params['object_type']
    ed = log['ocel:events'][target_event]
    objects = log['ocel:objects']
    target_object = [o for o in ed['ocel:omap']
                     if objects[o]['ocel:type'] == ot].pop()

    target_feature = pd.DataFrame([{'object choice': target_object}])

    # create sublog
    sobjects = ed['ocel:omap'] | {target_object}
    sevents = oh.get_relevant_events(log, graph, sobjects, target_event)
    situation_sublog = oh.create_sublog(log, sobjects, sevents)

    return {'target_feature': target_feature,
            'situation_features': None,
            'sublog': situation_sublog,
            'situation_event': target_event
            }


def validate_object_property(log, graph, target_object, params=None):
    prop = params['property']
    return prop in log['ocel:objects'][target_object]['ocel:ovmap']


def object_property_target(log, graph, target_object, params=None):
    prop = params['property']
    node = graph.nodes[target_object]
    situation_event = node['object_events'][0]
    sobjects = set(graph.predecessors(target_object)) | {target_object}
    sevents = oh.get_relevant_events(log, graph, sobjects, situation_event)

    target_feature = pd.DataFrame([{f'object <{prop}>':
                                    log['ocel:objects'][target_object][prop]}])

    situation_sublog = oh.create_sublog(log, sobjects, sevents)

    return {'target_feature': target_feature,
            'situation_features': None,
            'sublog': situation_sublog,
            'situation_event': situation_event
            }


def validate_object_lifetime(log, graph, target_object, params=None):
    final_an = params['final_activities']
    return graph.nodes[target_object]['object_events'][-1] in final_an


def object_lifetime_target(log, graph, target_object, params=None):
    events = log['ocel:events']
    oe = graph.nodes[target_object]['object_events']
    start_time = events[oe[0]]['ocel:timestamp']
    end_time = events[oe[-1]]['ocel:timestamp']
    sobjects = {target_object}
    for e in oe:
        sobjects |= events[e]['ocel:omap']

    sevents = set()
    for o in sobjects:
        sevents |= graph.noes[o]['object_events']

    situation_log = oh.create_sublog(log, sobjects, sevents)

    target_feature = pd.DataFrame([{'object lifetime': end_time - start_time}])

    return {'target_feature': target_feature,
            'situation_features': None,
            'sublog': situation_log,
            'situation_event': oe[-1]
            }


def validate_object_relation(log, graph, target_object, params=None):
    pass


def object_relation_target(log, graph, target_object, params=None):
    pass


def validate_timeframe(log, graph, target, params=None):
    events = log['ocel:events']
    first_time = events[events.keys()[0]]['ocel:timestamp']
    last_time = events[events.keys()[-1]]['ocel:timestamp']

    # check if one of the times are between the log bounds
    return (first_time <= params['start_time'] < last_time) \
        or (first_time < params['end_time'] <= last_time)


def global_timeframe_target(log, graph, target, params):
    events = log['ocel:events']
    work_type = params['workload_type']
    oids = set()
    eids = []

    # collect all info to do with
    for e, v in events.items():
        if params['start_time'] <= v['ocel:timestamp'] <= params['end_time']:
            eids.append(e)
            oids.update(v['ocel:omap'])

    situation_log = oh.create_sublog(log, oids, eids)

    if work_type == 'objects':
        features = og.Object_Global(situation_log, graph)
        features.add_global_obj_type_count()
        target_feature = features._df
    elif work_type == 'events':
        features = None

    return {'target_feature': target_feature,
            'situation_feature': None,
            'sublog': situation_log,
            'situation_event': None
            }


def validate_event_property_unknown(log, graph, target_event, params=None):
    e_dict = log['ocel:events'][target_event]
    na = {'NaN', 'NA', '', None, math.nan, np.NAN}

    return e_dict['ocel:vmap'][params['target_prop']] in na


def event_property_unknown(log, graph, target_event, params=None):
    events = log['ocel:events']
    prop = params['target_prop']
    oids = events[target_event]['ocel:omap']
    target_feature = pd.DataFrame([{f'event <{prop}>': '???'}])
    eids = set()
    for o in oids:
        eids.update(graph.nodes[o]['object_events'])

    situation_log = oh.create_sublog(log, oids, eids)

    return {'target_feature': target_feature,
            'situation_feature': None,
            'sublog': situation_log,
            'situation_event': target_event
            }


def validate_object_property_unknown(log, graph, target_object, params=None):
    o_dict = log['ocel:objects'][target_object]
    na = {'NaN', 'NA', '', None, math.nan, np.NAN}

    return o_dict['ocel:ovmap'][params['target_prop']] in na


def object_property_unknown(log, graph, target_object, params=None):
    node = graph.nodes[target_object]
    prop = params['target_prop']
    situation_event = node['object_events'][0]
    sobjects = set(graph.predecessors(target_object)) | {target_object}
    sevents = oh.get_relevant_events(log, graph, sobjects,
                                     oh.get_last_event(log))

    target_feature = pd.DataFrame([{f'object <{prop}>': '???'}])

    situation_log = oh.create_sublog(log, sobjects, sevents)

    return {'target_feature': target_feature,
            'situation_feature': None,
            'sublog': situation_log,
            'situation_event': situation_event
            }


def validate_object_missing_an(log, graph, target_object, params=None):
    node = graph.nodes[target_object]
    mia_activities = params['required_an']
    oan = set(oh.get_an_trace(log, node['object_events']))

    return mia_activities not in oan


def object_missing_an(log, graph, target_object, params=None):
    events = log['ocel:events']
    oe = graph.nodes[target_object]['object_events']
    mia_an = params['required_an'] - set(oh.get_an_trace(log, oe))
    sevents = oe
    sobjects = set()
    for e in sevents:
        sobjects.update(events[e]['ocel:omap'])

    situation_log = oh.create_sublog(log, sobjects, sevents)
    target_feature = pd.DataFrame([{'missing an': mia_an}])

    return {'target_feature': target_feature,
            'situation_feature': None,
            'sublog': situation_log,
            'situation_event': None
            }


def validate_event_missing_ot(log, graph, target_event, params=None):
    objects = log['ocel:objects']
    mia_ot = params['required_ot']  # dict with ot as key, count as value
    e_omap = Counter([objects[o]['ocel:type']
                      for o in log['ocel:events']['ocel:omap']])

    for k, v in mia_ot.items():
        if e_omap[k] < v:
            return True

    return False


def event_missing_ot(log, graph, target_event, params=None):
    events = log['ocel:events']
    objects = log['ocel:objects']

    sobjects = events[target_event]['ocel:omap']
    o_types = Counter([objects[o]['ocel:type'] for o in sobjects])
    sevents = set()
    missing = params['required_ot'] - o_types

    for o in sobjects:
        # add events
        sevents.update(graph.nodes[o]['object_events'])

    situation_log = oh.create_sublog(log, sobjects, sevents)
    target_feature = pd.DataFrame([{'missing ot': missing}])

    return {'target_feature': target_feature,
            'situation_feature': None,
            'sublog': situation_log,
            'situation_event': target_event
            }


def validate_object_missing_reachable_ot(log, graph, target_object, params):
    objects = log['ocel:objects']
    o_next = nx.descendants(graph, target_object)
    o_types = Counter([objects[o]['ocel:type'] for o in o_next])

    return params['required_ot'] - o_types


def object_missing_reachable_ot(log, graph, target_object, params):
    events = log['ocel:events']
    objects = log['ocel:objects']
    o_next = nx.descendants(graph, target_object)
    o_types = Counter([objects[o]['ocel:type'] for o in o_next])
    sevents = graph.nodes[target_object]['object_events']
    sobjects = o_next
    for e in sevents:
        sobjects.update(events[e]['ocel:omap'])

    missing = params['required_ot'] - o_types
    target_feature = pd.DataFrame([{'missing reachable ot': missing}])

    situation_log = oh.create_sublog(log, sobjects, sevents)

    return {'target_feature': target_feature,
            'situation_feature': None,
            'sublog': situation_log,
            'situation_event': None
            }


def validate_event_missing_relation(log, graph, target_event, params):
    e_dict = log['ocel:events'][target_event]
    req_rel = params['required_rel']
    rel_count = Counter()
    for o1, o2 in combinations(e_dict['ocel:omap'], 2):
        if o2 in graph[o1]:
            for rel, ev in graph[o1][o2].items():
                if target_event in ev:
                    rel_count.update([rel])
        if o1 in graph[o2]:
            for rel, ev in graph[o2][o1].items():
                if target_event in ev:
                    rel_count.update([rel])

    return req_rel - rel_count


def event_missing_relation(log, graph, target_event, params):
    ed = log['ocel:events'][target_event]

    sobjects = ed['ocel:omap']
    sevents = oh.get_relevant_events(log, graph, sobjects,
                                     oh.get_last_event(log))
    situation_sublog = oh.create_sublog(log, sobjects, sevents)

    # get relation count difference
    req_rel = params['required_rel']
    rel_count = Counter()
    for o1, o2 in combinations(ed['ocel:omap'], 2):
        if o2 in graph[o1]:
            for rel, ev in graph[o1][o2].items():
                if target_event in ev:
                    rel_count.update([rel])
        if o1 in graph[o2]:
            for rel, ev in graph[o2][o1].items():
                if target_event in ev:
                    rel_count.update([rel])

    target_feature = pd.DataFrame([{'missing relations': req_rel - rel_count}])

    return {'target_feature': target_feature,
            'situation_features': None,
            'sublog': situation_sublog,
            'situation_event': target_event
            }


# format validator function, target feature generator, required props
class Targets(Enum):
    # Event Based -> require an event
    EVENTCHOICE = (validate_event_choice_target,
                   event_choice_target,
                   ['activities'])
    EVENTPROPERTY = (validate_event_property_target,
                     event_property_target,
                     ['property'])
    EVENT_PROPERTY_UNKNOWN = (validate_event_property_unknown,
                              event_property_unknown,
                              ['target_prop'])
    EVENTWAIT = (validate_event_wait,
                 event_wait_target,
                 [])
    EVENT_DURATION = (validate_event_duration,
                      event_duration_target,
                      [])
    EVENTOBJECTCHOICE = (validate_object_choice,
                         object_choice_target,
                         ['activities', 'object_type'])

    EVENT_MISSING_REL = (validate_event_missing_relation,
                         event_missing_relation,
                         ['required_rel'])

    EVENT_MISSING_OT = (validate_event_missing_ot,
                        event_missing_ot,
                        ['required_ot'])
    # Object Based -> require an object
    OBJECTPROPERTY = (validate_object_property,
                      object_property_target,
                      ['property'])
    OBJECTPROPERTYUNKNOWN = (validate_object_property_unknown,
                             object_property_unknown,
                             [])

    OBJECT_MISSING_ACTIVITY = (validate_object_missing_an,
                               object_missing_an,
                               ['required_an'])
    OBJECTLIFETIME = (validate_object_lifetime,
                      object_lifetime_target,
                      ['final_activities'])
    OBJECT_MISSING_REACHABLE_OT = (validate_object_missing_reachable_ot,
                                   object_missing_reachable_ot,
                                   ['required_ot'])

    # local based -> require target object (based on its lineage/locality)
    LINEAGE_PROPERTY_OP = ()
    LINEAGE_MISSING_OT = ()

    # Global Based -> Require a timeframe
    TIMEWORKLOAD = (validate_timeframe,
                    global_timeframe_target,
                    ['start_time', 'end_time', 'workload_type'])
    TIMEOPPROPERTY = ()


def create_situations(log, graph, targets,
                      target_feature=Targets.EVENTCHOICE, params=None):

    # expand the target
    validator = target_feature.value[0]
    target_creator = target_feature.value[1]
    req_params = target_feature.value[2]
    # check param properties
    if req_params and not params:
        print('Please provide parameters',
              f'{req_params} to use {Targets(target_feature).name}.')
        return
    elif params:
        not_contained = []
        for p in req_params:
            if p not in params:
                not_contained.append(p)
        if not_contained:
            print('Please additionally provide parameters',
                  f'{req_params} to use {Targets(target_feature).name}.')
            return

    # generate the situations
    situations = []
    invalid_targets = set()
    for t in targets:
        if validator(log, graph, t, params):
            target_dict = target_creator(log, graph, t, params)
            situations.append(Situation(log, graph, target_dict))
        else:
            invalid_targets.add(t)

    return situations
