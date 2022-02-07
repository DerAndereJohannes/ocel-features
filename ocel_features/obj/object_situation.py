import pandas as pd
import numpy as np
from copy import copy
from networkx.algorithms.shortest_paths import shortest_path, \
    all_shortest_paths
from ocel_features.util.multigraph import relations_to_relnames
from ocel_features.util.local_helper import obj_relationship_localities
import ocel_features.obj.object_point as op
import ocel_features.obj.object_global as og
# import ocel_features.util.ocel_helper as oh


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
