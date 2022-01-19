import ocel
import networkx as nx
from enum import Enum

_RELATION_DELIMITER = '2'
_SHORT_LENGTH = 3

def create_multi_graph(log, relations):
    # remove duplicate functions
    #rels = {rel.value[1] for rel in relations}

    ocel_events = ocel.get_events(log)
    ocel_objects = ocel.get_objects(log)
    ids_checked = set()
    net = nx.DiGraph()

    for event_id in ocel_events:
        event = ocel_events[event_id]

        for oid in event['ocel:omap']:
            # add first occured time and event attribute to the node
            if oid not in ids_checked:
                ids_checked.add(oid)
                net.add_node(oid)
                net.nodes[oid]['type'] = ocel_objects[oid]['ocel:type']
                net.nodes[oid]['object_events'] = list()
            net.nodes[oid]['object_events'].append(event)

    for event_id in ocel_events:
        event = ocel_events[event_id]

        for oid in event['ocel:omap']:
            # add all new edges between selected
            for oid2 in event['ocel:omap']:
                if oid is not oid2:
                    for rel in relations:
                        execute_relations(net, event, oid, oid2, rel)

    return net


# HELPER FUNCTIONS
def same_index_event(net, id_1, id_2, index):
    return net.nodes[id_1]['object_events'][index] is \
        net.nodes[id_2]['object_events'][index]


def same_index_time(net, id_1, id_2, index):
    return net.nodes[id_1]['object_events'][index]['ocel:timestamp'] == \
        net.nodes[id_2]['object_events'][index]['ocel:timestamp']


def get_younger_obj(net, id_1, id_2):
    obj1_birth = net.nodes[id_1]['object_events'][0]['ocel:timestamp']
    obj2_birth = net.nodes[id_2]['object_events'][0]['ocel:timestamp']
    if obj1_birth < obj2_birth:
        return id_1
    else:
        return id_2


def has_init_node(net, id_1, id_2, event):
    return net.nodes[id_1]['object_events'][0] is event \
        or net.nodes[id_2]['object_events'][0] is event


def has_death_and_birth(net, id_1, id_2):
    if net.nodes[id_1]['object_events'][0] is \
            net.nodes[id_2]['object_events'][-1]:
        return id_2, id_1
    elif net.nodes[id_1]['object_events'][-1] is \
            net.nodes[id_2]['object_events'][0]:
        return id_1, id_2
    else:
        return None, None


def is_descendant(net, source, target):
    return target in net.nodes[source]['descendant']


def execute_relations(net, event, src, tar, rel):
    rel_name = Relations(rel).name
    rel1 = rel2 = None
    if _RELATION_DELIMITER in rel_name:
        rel1, rel2 = split_multi_rel(rel_name)
    else:
        rel1 = rel_name
    
    rel.value[0](net, event, src, tar, (rel1, rel2))


def split_multi_rel(rel_name):
    return rel_name.split(_RELATION_DELIMITER)


def relations_to_relnames(rels):
    rel_set = set()

    for rel in rels:
        rel_name = Relations(rel).name
        if _RELATION_DELIMITER in rel_name:
            rel_set |= set(split_multi_rel(rel_name))
        else:
            rel_set.add(rel_name)

    return rel_set



def add_directed_edge(net, event, src, tar, rel_names):
    if not net.has_edge(src,tar):
        net.add_edge(src,tar)

    rel = rel_names[0]

    if not rel in net[src][tar]:
        net[src][tar][rel] = [event]
    else:
        net[src][tar][rel].append(event)


def add_undirected_edge(net, event, src, tar, rel_names):
    if not (net.has_edge(src,tar) or net.has_edge(tar, src)):
        net.add_edge(src,tar)
        net.add_edge(tar, src)

    rel1, rel2 = rel_names
    if rel2 is None:
        rel2 = rel1

    # add the event to both of the relations
    if not rel1 in net[src][tar]:
        net[src][tar][rel1] = [event]
    else:
        net[src][tar][rel1].append(event)

    if not rel2 in net[tar][src]:
        net[tar][src][rel2] = [event]
    else:
        net[tar][src][rel2].append(event)


# RELATIONSHIP TYPES
def add_interaction(net, event, src, tar, rel_names):
    add_undirected_edge(net, event, src, tar, rel_names)
    # net.add_edge(src, tar, interacts=True)
    # net.add_edge(tar, src, interacts=True)


def add_descendants(net, event, src, tar, rel_names):
    if has_init_node(net, src, tar, event) \
            and not same_index_event(net, src, tar, 0):
        if src == get_younger_obj(net, src, tar):
            add_directed_edge(net, event, src, tar, rel_names)
            # net.add_edge(src, tar, descendant=True)
        else:
            add_directed_edge(net, event, tar, src, rel_names)
            # net.add_edge(tar, src, descendant=True)


def add_ancestors(net, event, src, tar, rel_names):
    if has_init_node(net, src, tar, event) \
            and not same_index_event(net, src, tar, 0):
        if src == get_younger_obj(net, src, tar):
            add_directed_edge(net, event, tar, src, rel_names)
        else:
            add_directed_edge(net, event, src, tar, rel_names)


def add_lineage(net, event, src, tar, rel_names):
    if has_init_node(net, src, tar, event) \
            and not same_index_event(net, src, tar, 0):
        if src == get_younger_obj(net, src, tar):
            add_undirected_edge(net, event, tar, src, rel_names)
        else:
            add_undirected_edge(net, event, src, tar, rel_names)


def add_cobirth(net, event, src, tar, rel_names):
    if same_index_event(net, src, tar, 0):
        add_undirected_edge(net, event, src, tar, rel_names)


def add_codeath(net, event, src, tar, rel_names):
    if same_index_event(net, src, tar, -1):
        add_undirected_edge(net, event, src, tar, rel_names)


def add_colife(net, event, src, tar, rel_names):
    src_events = net.nodes[src]['object_events']
    tar_events = net.nodes[tar]['object_events']

    if src_events == tar_events:
        add_undirected_edge(net, event, src, tar, rel_names)


def add_merge(net, event, src, tar, rel_names):
    if net.nodes[src]['type'] == net.nodes[tar]['type']:
        src_events = net.nodes[src]['object_events']
        tar_events = net.nodes[tar]['object_events']
        if src_events[-1] in tar_events[:-1]:
            add_directed_edge(net, event, src, tar, rel_names)
            # net.add_edge(src, tar, merge=True)


def add_consumes(net, event, src, tar, rel_names):
    if net.nodes[src]['type'] != net.nodes[tar]['type']:
        src_events = net.nodes[src]['object_events']
        tar_events = net.nodes[tar]['object_events']

        if tar_events[-1] in src_events[:-1]:
            add_directed_edge(net, event, src, tar, rel_names)
            # net.add_edge(src, tar, consumes=True)


def add_inheritance(net, event, src, tar, rel_names):
    if has_init_node(net, src, tar, event):
        death, birth = has_death_and_birth(net, src, tar)
        if death is not None:
            add_directed_edge(net, event, death, birth, rel_names)


def add_minion(net, event, src, tar, rel_names):
    src_events = net.nodes[src]['object_events']
    tar_events = net.nodes[tar]['object_events']

    if all(x in src_events for x in tar_events) \
       and len(tar_events) < len(src_events):
        add_directed_edge(net, event, tar, src, rel_names)
        # net.add_edge(tar, src, minion=True)


# Relationship requires a different format to be efficient
def add_peeler(net, event, src, tar, rel_names):
    src_events = net.nodes[src]['object_events']

    for e in src_events:
        if tar in e['ocel:omap'] and not e['ocel:omap'] == {src, tar}:
            return

    add_undirected_edge(net, event, src, tar, rel_names)
    # net.add_edge(src, tar, peeler=True)
    # net.add_edge(src, tar, peeler=True)


# MAIN FUNCTIONS
class Relations(Enum):
    INTERACTS = (add_interaction, )
    DESCENDANTS = (add_descendants, )
    ANCESTORS = (add_ancestors, )
    ANCESTORS2DESCENDANTS = (add_lineage, )
    COBIRTH = (add_cobirth, )
    CODEATH = (add_codeath, )
    COLIFE = (add_colife, )
    MERGE = (add_merge, )
    INHERITANCE = (add_inheritance, )
    MINION = (add_minion, )
    PEELER = (add_peeler, )
    CONSUMES = (add_consumes, )


def relation_shorthand(rel):
    if isinstance(rel, str):
        return rel[:_SHORT_LENGTH]
    elif isinstance(rel, Relations):
        return Relation(rel.value).name[:_SHORT_LENGTH]
    else:
        return None


def create_object_centric_graph(log, relations=None):
    if relations is None:
        relations = [Relations.ANCESTORS2DESCENDANTS]

    return create_multi_graph(log, relations)
