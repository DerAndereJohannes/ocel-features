import networkx as nx
from itertools import chain
from collections import Counter


def get_direct_relations_count(net, src):
    lists_of_relations = [list(k.keys()) for k in net[src].values()]
    return Counter(chain.from_iterable(lists_of_relations))


def get_direct_relations_type_tuple(net, src, obj_list):
    return [(obj_list[k]['ocel:type'], v) for k, v in net[src].items()]


def get_obj_relation_trees(net, events, rels):
    rel_trees = {n: None for n in net.nodes}
    for k in rel_trees:
        rel_trees[k] = {rel: {k} for rel in rels}
        rel_trees[k]['contains'] = set()

    return rel_trees


def has_multi_relation_ot(net, node, rel_name, ot):
    es = set()
    # check if there are multiple of OT
    [es.update(net.nodes[n2][rel_name]) for n2 in nx.neighbors(net, node)
     if net.nodes[n2]['type'] == ot]

    return len(es) > 1


def get_graph_roots(net):
    return [n for n, d in net.in_degree() if d == 0]


def decompose_multi_relations(net):
    from copy import deepcopy
    net = deepcopy(net)
    cuts = set()
    # gather all split points
    for o in net.nodes():
        out = set(nx.neighbors(net, o))
        if len(out) > 1:
            cuts.update({(o, o2) for o2 in out})

    # Create subgraphs with splitpoints
    print(cuts)
    net.remove_edges_from(cuts)

    return nx.weakly_connected_components(net)
