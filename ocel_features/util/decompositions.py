import numpy as np
import networkx as nx
from enum import Enum
from itertools import combinations, product


def get_timestamp(log, event):
    return log['ocel:events'][event]['ocel:timestamp']


def get_event_separation(graph, check_degree=0):
    e_d = {}
    out_deg = graph.out_degree()

    # gather event and edge data for all obj which have more than one out arc
    # (ie. with a potential to violate decomp)
    for u, v, data in graph.edges(data=True):
        if out_deg[u] > check_degree:
            if 'DESCENDANTS' not in data:
                continue
            curr_e = next(iter(data['DESCENDANTS']))
            if u not in e_d:
                e_d[u] = dict()

            if curr_e not in e_d[u]:
                e_d[u][curr_e] = set()

            e_d[u][curr_e].add(v)

    return e_d


# returns set of edges that would have to be removed for the
# decomposition to work
def pointwise_relation_decomp(log, graph, params):
    # if no o_types are specified, use all object types
    if not params:
        o_types = set(log['ocel:global-log']['ocel:object-types'])
    else:
        o_types = params

    objects = log['ocel:objects']
    violations = set()
    e_d = get_event_separation(graph, 1)

    # go through each found potential object and check if it violates
    for src, v in e_d.items():
        # if object participates in more than one creation event
        if len(v) > 1:
            type_check = {ot: set() for ot in o_types}

            # check all events to see which o types they have
            for event, out in v.items():
                for o in out:
                    o_type = objects[o]['ocel:type']
                    if o_type in o_types:
                        type_check[o_type].add(event)

            # add all edges to do with event to violation list
            for ot, evs in type_check.items():
                if len(evs) > 1:
                    for e in evs:
                        violations.update({(src, tar) for tar in v[e]})

    return violations


def change_relation_matrix(rel_mat, src, tar, value):
    if value > rel_mat[src][tar]:
        rel_mat[src][tar] = min(value, 2)


def generate_global_relation_matrix(log, graph):
    # if no o_types are specified, use all object types
    o_types = {ot: i for i, ot in enumerate(log['ocel:global-log']
                                               ['ocel:object-types'])}

    objects = log['ocel:objects']
    rel_mat = np.zeros((len(o_types), len(o_types)), dtype=np.ushort)
    e_d = get_event_separation(graph, 0)

    # go through each found potential object and check if it violates
    for src, v in e_d.items():
        src_type = objects[src]['ocel:type']
        # if object participates in more than one creation event
        type_check = {ot: set() for ot in o_types}

        # check all events to see which o types they have
        for event, out in v.items():
            for o in out:
                o_type = objects[o]['ocel:type']
                if o_type in o_types:
                    type_check[o_type].add(event)

        # add all edges to do with event to violation list
        for ot, evs in type_check.items():
            change_relation_matrix(rel_mat, o_types[src_type],
                                   o_types[ot], len(evs))
    return rel_mat, o_types


def matrix_to_violations(log, graph, rel_mat):
    o_types = {ot: i for i, ot in enumerate(log['ocel:global-log']
                                               ['ocel:object-types'])}
    objects = log['ocel:objects']
    to_remove_edges = set()

    for src, tar in graph.edges():
        src_i = o_types[objects[src]['ocel:type']]
        tar_i = o_types[objects[tar]['ocel:type']]
        if rel_mat[src_i][tar_i] > 1:
            to_remove_edges.add((src, tar))

    return to_remove_edges


def global_relation_decomp(log, graph, params):
    relation_matrix, _ = generate_global_relation_matrix(log, graph)
    print(relation_matrix)
    return matrix_to_violations(log, graph, relation_matrix)


class Decompositions(Enum):
    GLOBAL_RELATION = (global_relation_decomp,)
    POINTWISE_RELATION = (pointwise_relation_decomp,)


def decomp_graph_relations(log, graph,
                           decomp=Decompositions.GLOBAL_RELATION, params=None):
    graph = graph.copy()
    violations = decomp.value[0](log, graph, params)
    graph.remove_edges_from(violations)
    return graph


def get_disconnected_subgraphs(graph):
    subgraph_nodes = list(nx.weakly_connected_components(graph))
    return [graph.subgraph(snodes) for snodes in subgraph_nodes]


def get_event_relation_dict(log, graph):
    relation_matrix, o_types = generate_global_relation_matrix(log, graph)
    rel = {0: 'None', 1: 'Single', 2: 'Many'}
    rel_dict = {i: {j: 0} for i, j in product(o_types.keys(), o_types.keys())}

    for ot, i in o_types.items():
        for ot2, j in o_types.items():
            rel_dict[ot][ot2] = rel[relation_matrix[i][j]]

    return rel_dict


def summarize_relations(rel_dict):
    indicies = list(combinations(rel_dict.keys(), 2)) \
                    + [(k, k) for k in rel_dict.keys()]
    for i, j in indicies:
        print(f'{i} -> {j}: {rel_dict[i][j]}\n {j} -> {i}: {rel_dict[j][i]}')


def create_subgraph_from_edges(graph, edges):
    pass


def create_subgraph_from_nodes(graph, nodes):
    pass
