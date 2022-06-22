import networkx as nx
from itertools import combinations_with_replacement
from ocel_features.util.multigraph import create_object_centric_graph

lockable = ['DESCENDANTS', 'COBIRTH', 'MERGE', 'INHERITANCE', 'SPLIT']
# relations lock objects together, if from same event put in same tuple
# to compute when relations unlock, find out last events between object types
# and unlock on those objects (threshold?)


def get_unlocking_events(log):
    ots = log['ocel:global-log']['ocel:object-types']
    ot_combinations = combinations_with_replacement(ots, 2)
    objects, events = log['ocel:objects'], log['ocel:events']

    # ot_combo_end =
    seen_obj = set()
    # pointwise
    start_stop = {oid: {ot: {'start': {}, 'stop': {}} for ot in ots} for oid in objects}
    for e in events:
        pass

    return ot_combo_end


def create_locking_graph(log):
    in_graph = create_object_centric_graph(log, relations=lockable)
    locking_graph = nx.DiGraph()

    return locking_graph
