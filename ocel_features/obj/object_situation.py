import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from enum import Enum
from copy import copy
from networkx.algorithms.shortest_paths import shortest_path, \
    all_shortest_paths
from ocel_features.util.multigraph import relations_to_relnames, \
    rel_subgraph, Relations
from ocel_features.util.local_helper import obj_relationship_localities
from ocel_features.util.data_organization import Operators
import ocel_features.obj.object_point as op
import ocel_features.obj.object_global as og
# import ocel_features.util.ocel_helper as oh


class Object_Situation:
    pass


class Event_Situation:
    pass


class Object_Situation_Locality_OT:
    def __init__(self, log, graph, localities, oid, situation):
        self._log = log
        self._graph = graph
        # self._log, self._graph = oh.create_subproblem(log, graph,
        #                                               situation['objects'],
        #                                               situation['events'])
        self._localities = localities  # only of the included nodes
        self._source = situation['situation_source']
        self._target = situation['situation_target']
        self._target_event = situation['situation_event']
        self._situation = situation  # one one situation

        self._df = pd.DataFrame()

    def get_obj_attributes(self):
        for o in self._situation['objects']:
            for k, v in self._log['ocel:objects'][o]['ocel:ovmap'].items():
                ot = self._log['ocel:objects'][o]['ocel:type']
                key = f'att:{ot}:{o}:{k}'
                self._df[key] = [v]

    def get_ot_attributes(self, ot: set = None):
        if not isinstance(ot, set):
            ot = set(self._log['ocel:global-log']['ocel:object-types'])
        for o in self._situation['objects']:
            o_ot = self._log['ocel:objects'][o]['ocel:type']
            if o_ot in ot:
                for k, v in self._log['ocel:objects'][o]['ocel:ovmap'].items():
                    key = f'att:{ot}:{o}:{k}'
                    self._df[key] = [v]

    def get_agg_ot_attributes(self, ot: set = None):
        # only goes over values
        if not isinstance(ot, set):
            ot = set(self._log['ocel:global-log']['ocel:object-types'])
        curr_col = copy(self._df.columns)
        for o in self._situation['objects']:
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
        for e in self._situation['events']:
            for k, v in self._log['ocel:events'][e]['ocel:vmap'].items():
                an = self._log['ocel:events'][e]['ocel:activity']
                key = f'att:{an}:{e}:{k}'
                self._df[key] = [v]

    def get_agg_activity_attributes(self):
        # only goes over values
        curr_col = copy(self._df.columns)
        for e in self._situation['events']:
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


# FILTERING TO GET GOOD SITUATION PREFIXES ######
def filter_direct_activity_involvement(log, graph, oids: set, ans: set):
    events = log['ocel:events']
    obj_situations = {}
    for o in oids:
        od = {'oid': o, 'situations': []}
        ev_ids = set()
        oids = set()
        for i, ev in enumerate(graph.nodes[o]['object_events']):
            ev_ids.add(ev)
            oids.update(events[ev]['ocel:omap'])
            if events[ev]['ocel:activity'] in ans:
                situation = {'event': ev, 'index': i,
                             'events': copy(ev_ids), 'objects': copy(oids)}
                od['situations'].append(situation)

        if not od['situations']:
            od = None

        obj_situations[o] = od

    return obj_situations


# FILTERING TO GET GOOD SITUATION PREFIXES ######
def filter_event_ot_involvement(log, graph, oids: set, ot: set):
    events = log['ocel:events']
    objects = log['ocel:objects']
    obj_situations = {}
    for o in oids:
        od = {'oid': o, 'situations': []}
        ev_ids = set()
        oids = set()
        for i, ev in enumerate(graph.nodes[o]['object_events']):
            ev_ids.add(ev)
            oids.update(events[ev]['ocel:omap'])
            omap = events[ev]['ocel:omap']
            for o2 in omap:
                o2_type = objects[o2]['ocel:type']
                if o2_type in ot:
                    situation = {'oid': o2, 'type': o2_type,
                                 'event': ev, 'index': i,
                                 'events': copy(ev_ids)}
                    objs = set()
                    for e in ev_ids:
                        objs |= events[e]['ocel:omap']
                    situation['objects'] = objs
                    od['situations'].append(situation)

        if not od['situations']:
            od = None

        obj_situations[o] = od

    return obj_situations


def get_oids_specific_event_involvement(log, graph, oids, eids):
    rtn_set = set()
    for o in oids:
        if eids.issubset(graph.nodes[o]['object_events']):
            rtn_set.add(o)

    return rtn_set


def filter_locality_ot_involvement(log, graph, localities, oids, ot, rels):
    rels = relations_to_relnames(rels)
    obj_situations = {}
    events = log['ocel:events']

    for o in oids:
        od = {'src': o, 'situations': []}
        for rel in rels:
            if localities[o][rel]:
                o_set, o_tree = localities[o][rel]
                for o2 in o_set:
                    if log['ocel:objects'][o2]['ocel:type'] in ot:
                        path = shortest_path(o_tree, o, o2)
                        situation = {'relation': rel,
                                     'path': path}
                        evs = []
                        objs = set()
                        switch_ev = graph.nodes[o]['object_events'][0]
                        for i, oid in enumerate(path[:-1]):
                            oe = graph.nodes[oid]['object_events']
                            start_index = oe.index(switch_ev)
                            for e in oe[start_index:]:
                                if path[i+1] in events[e]['ocel:omap']:
                                    switch_ev = e
                                    break
                                else:
                                    evs.append(e)
                                    objs |= events[e]['ocel:omap']

                        situation['objects'] = objs
                        situation['events'] = evs
                        situation['situation_event'] = switch_ev
                        situation['situation_source'] = o
                        situation['situation_target'] = o2
                        od['situations'].append(situation)

        obj_situations[o] = od

    return obj_situations


def filter_global_ot_involvement(log, graph, oids, ot, rels):
    localities = obj_relationship_localities(graph, rels)
    rels = relations_to_relnames(rels)
    obj_situations = {}

    for o in oids:
        od = {'oid': o, 'situations': []}
        for rel in rels:
            o_set, o_tree = localities[o][rel]
            for o2 in o_set:
                if log['ocel:objects'][o2]['ocel:type'] == ot:
                    situation = {'o2': o2,
                                 'paths': all_shortest_paths(graph, o, o2)}
                    situation['objects'] = 0
                    situation['events'] = 0
                    od['situations'].append(situation)

    return obj_situations


def filter_local_relation_ot_involvement(log, graph, oids, ot, rels):
    rtn_set = set()
    localities = obj_relationship_localities(graph, rels)
    rels = relations_to_relnames(rels)

    for o in oids:
        for rel in rels:
            for o2 in localities[o][rel]:
                if log['ocel:objects'][o2]['ocel:type'] == ot:
                    rtn_set.add(o)
                    break

    return rtn_set


def filter_target_situations(log, graph, localities, targets, rels):
    rels = relations_to_relnames(rels)
    obj_situations = {}
    events = log['ocel:events']

    # setup
    igraph = rel_subgraph(graph, rels).reverse()

    for o in targets:
        situation = {'target': o}
        situation['target_event'] = graph.nodes[o]['object_events'][0]
        situation['objects'] = nx.descendants(igraph, o) | {o}
        situation['graph'] = igraph.subgraph(situation['objects'])
        sit_events = {situation['target_event']}
        last_time = events[situation['target_event']]['ocel:timestamp']

        for o2 in situation['objects']:
            sit_events.update({e for e in graph.nodes[o2]['object_events']
                               if events[e]['ocel:timestamp'] <= last_time})
        situation['events'] = sit_events

        obj_situations[o] = situation

    return obj_situations


def create_situations_obj_rel_ot(log, graph, localities,
                                 src: str, ot: str, rels: set):
    d = []
    src = {src} if isinstance(src, str) else src
    ot = {ot} if isinstance(src, str) else ot

    # get the individual situations
    situations = filter_locality_ot_involvement(log, graph, localities,
                                                src, ot, rels)

    # create a df object for every situation
    for s in situations[src.pop()]['situations']:
        d.append(Object_Situation_Locality_OT(log, graph, localities, src, s))
    return d


def filter_object_situation(log, igraph, targets, rels, an=None):

    obj_situations = {}
    events = log['ocel:events']

    if an:
        for o in targets:
            an_e = [e for e in igraph.nodes[o]['object_events']
                    if events[e]['ocel:activity'] in an]

            # check if the objects events contain the needed activity name(s)
            if an_e:
                situation = {'target_object': o,
                             'target_type': igraph.nodes[o]['type']}
                situation['target_events'] = an_e
                situation['objects'] = nx.descendants(igraph, o)
                situation['graph'] = igraph.subgraph(situation['objects'])
                obj_situations[o] = situation
    else:
        for o in targets:
            situation = {'target_object': o,
                         'target_type': igraph.nodes[o]['type']}
            situation['target_event'] = igraph.nodes[o]['object_events'][0]
            situation['objects'] = nx.descendants(igraph, o)
            situation['graph'] = igraph.subgraph(situation['objects'])
            obj_situations[o] = situation

    return obj_situations


def create_event_situations(log, graph, an=None):
    e_situations = []
    events = log['ocel:events']

    for e, e_dict in events:
        if e_dict['ocel:activity'] in an:
            e_sit = {'target_event': e,
                     'target_activity': e_dict['ocel:activity']}

            e_situations.append(Event_Situation(log, e_sit))

    return e_situations


def create_event_situation(e, e_dict, graph):
    return {'target_event': e,
            'target_activity': e_dict['ocel:activity'],
            'objects': e_dict['ocel:omap'],
            'graph': graph}


def create_object_situations(log, graph, targets,
                             rels={Relations.INTERACTS}, an=None):
    output = []
    targets = {targets} if isinstance(targets, str) else targets
    rels = relations_to_relnames(rels)
    igraph = rel_subgraph(graph, rels)

    situations = filter_object_situation(log, igraph, targets, rels, an)

    if an:
        for o, s in situations:
            for e in s['target_events']:
                output.append(Object_Situation(log, s))
    else:
        output = [Object_Situation(log, s) for s in situations.values()]

    return output


class Situation:
    def __init__(self, log, graph, params):
        self._log = log
        self._graph = graph
        self._objects = set(params['objects'])
        self._maxevent = params['event']
        self._maxtime = log['ocel:events'][params['event']]['ocel:timestamp']

        self._df = pd.DataFrame()
        self._targetdf = pd.DataFrame({
            params['target_name']: [params['target_feature']]})

        self._events = set()
        o_add = set()
        for o in self._objects:
            for e in graph.nodes[o]['object_events']:
                if log['ocel:events'][e]['ocel:timestamp'] < self._maxtime:
                    self._events.add(e)
                    o_add.update(log['ocel:events'][e]['ocel:omap'])
        self._objects.update(o_add)

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


def validate_event_choice_target(log, graph, target_event, params=None):
    an = params['activities']
    return log['ocel:events'][target_event]['ocel:activity'] in an


def event_choice_target(log, graph, target_event, params=None):
    ed = log['ocel:events'][target_event]
    return {'target_name': 'event choice',
            'target_feature': ed['ocel:activity'],
            'objects': ed['ocel:omap'],
            'event': target_event}


def validate_event_property_target(log, graph, target_event, params=None):
    prop = params['property']
    return prop in log['ocel:events'][target_event]['ocel:vmap']


def event_property_target(log, graph, target_event, params=None):
    prop = params['property']
    ed = log['ocel:events'][target_event]
    return {'target_name': f'event <{prop}>',
            'target_feature': ed['ocel:vmap'][prop],
            'objects': ed['ocel:omap'],
            'event': target_event}


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

    now = datetime.now()
    oldest_time = now
    for o in ed['ocel:omap']:
        oe = graph.nodes[o]['object_events']
        if target_event != oe[0]:
            i = oe.index(target_event) - 1
            e_time = ev[oe[i]]['ocel:timestamp']
            if e_time < oldest_time:
                oldest_time = e_time

    return {'target_name': 'wait until event',
            'target_feature': ed['ocel:timestamp'] - oldest_time,
            'objects': ed['ocel:omap'],
            'event': target_event}


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

    return {'target_name': 'object choice',
            'target_feature': target_object,
            'objects': ed['ocel:omap'] | {target_object},
            'event': target_event}


def validate_object_property(log, graph, target_object, params=None):
    prop = params['property']
    return prop in log['ocel:objects'][target_object]['ocel:ovmap']


def object_property_target(log, graph, target_object, params=None):
    prop = params['property']
    node = graph.nodes[target_object]
    interactedo = set(graph.predecessors(target_object)) | {target_object}

    return {'target_name': f'object <{prop}>',
            'target_feature': log['ocel:objects'][target_object][prop],
            'objects': interactedo,
            'event': node['object_events'][0]}


def validate_object_lifetime(log, graph, target_object, params=None):
    final_an = params['final_activities']
    return graph.nodes[target_object]['object_events'][-1] in final_an


def object_lifetime_target(log, graph, target_object, params=None):
    events = log['ocel:events']
    oe = graph.nodes[target_object]['object_events']
    start_time = events[oe[0]]['ocel:timestamp']
    end_time = events[oe[-1]]['ocel:timestamp']
    interactedo = {target_object}
    for e in oe:
        interactedo |= events[e]['ocel:omap']

    return {'target_name': 'object lifetime',
            'target_feature': end_time - start_time,
            'objects': interactedo,
            'event': oe[-1]}


def validate_object_relation(log, graph, target_object):
    pass


def object_relation_target(log, graph, target_object):
    pass


# format validator function, target feature generator, required props
class Targets(Enum):
    # Event Based -> require an event
    EVENTCHOICE = (validate_event_choice_target,
                   event_choice_target,
                   ['activities'])
    EVENTPROPERTY = (validate_event_property_target,
                     event_property_target,
                     ['property'])
    EVENTWAIT = (validate_event_wait,
                 event_wait_target,
                 [])
    EVENTOBJECTCHOICE = (validate_object_choice,
                         object_choice_target,
                         ['activities', 'object_type'])
    # Object Based -> require an object
    OBJECTPROPERTY = (validate_object_property,
                      object_property_target,
                      ['property'])
    OBJECTLIFETIME = (validate_object_lifetime,
                      object_lifetime_target,
                      ['final_activities'])
    OBJECTRELATION = ()


def create_situations(log, graph, targets,
                      target_feature=Targets.EVENTCHOICE, params=None):

    # expand the target
    validator = target_feature.value[0]
    target_creator = target_feature.value[1]
    req_params = target_feature.value[2]
    # check properties
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
