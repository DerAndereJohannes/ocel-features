import pandas as pd
from copy import copy
from networkx.algorithms.shortest_paths import shortest_path, \
    all_shortest_paths
from ocel_features.util.multigraph import create_object_centric_graph, \
    relations_to_relnames
from ocel_features.util.local_helper import obj_relationship_localities


class Object_Situation:
    def __init__(self, log, oids=None, graph=None):
        self._log = log
        if graph:
            self._graph = graph
        else:
            self._graph = create_object_centric_graph(log)

        if not oids:
            oids = log['ocel:objects']

        self._df = pd.DataFrame({'oid': oids,
                                 'type': [log['ocel:objects'][x]['ocel:type']
                                          for x in oids]})
        self._op_log = []

        # 1. get all attributes based on object type AKA. duplicate OT is OK
        # 1.5. get global attributes to OE (object-point attributes)
        # 2. get all event and event attributes based on AN of the OE instance
        # connection to object global.
        # 3. gather all events of all objects
        # 4.


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


def get_oids_specific_event_involvement(log, graph, oids, eids):
    rtn_set = set()
    for o in oids:
        if eids.issubset(graph.nodes[o]['object_events']):
            rtn_set.add(o)

    return rtn_set


def filter_locality_ot_involvement(log, graph, oids, ot, rels):
    localities = obj_relationship_localities(graph, rels)
    rels = relations_to_relnames(rels)
    obj_situations = {}

    for o in oids:
        od = {'oid': o, 'situations': []}
        for rel in rels:
            o_set, o_tree = localities[o][rel]
            for o2 in o_set:
                if log['ocel:objects'][o2]['ocel:type'] in ot:
                    situation = {'o2': o2, 'relation': rel,
                                 'path': shortest_path(o_tree, o, o2)}
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
