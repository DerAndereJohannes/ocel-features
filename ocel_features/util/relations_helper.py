from itertools import chain
from collections import Counter


def get_direct_relations_count(net, src):
    lists_of_relations = [list(k.keys()) for k in net[src].values()]
    return Counter(chain.from_iterable(lists_of_relations))


def get_obj_relation_trees(net, events, rels):
    rel_trees = {n: None for n in net.nodes}
    for k in rel_trees:
        rel_trees[k] = {rel: {k} for rel in rels}
        rel_trees[k]['contains'] = set()

    return rel_trees
