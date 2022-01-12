from enum import Enum
import ocel
import networkx as nx


def create_multi_graph(log, relations):
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
                        rel(net, event, oid, oid2)

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


# RELATIONSHIP TYPES
def add_interaction(net, event, src, tar):
    net.add_edge(src, tar, interacts=True)
    net.add_edge(tar, src, interacts=True)


def add_descendants(net, event, src, tar):
    if has_init_node(net, src, tar, event) \
            and not same_index_event(net, src, tar, 0):
        if src == get_younger_obj(net, src, tar):
            net.add_edge(src, tar, descendant=True)
            net.add_edge(tar, src, ancestor=True)
        else:
            net.add_edge(src, tar, ancestor=True)
            net.add_edge(tar, src, descendant=True)


def add_cobirth(net, event, src, tar):
    if same_index_event(net, src, tar, 0):
        net.add_edge(src, tar, cobirth=True)
        net.add_edge(tar, src, cobirth=True)


def add_codeath(net, event, src, tar):
    if same_index_event(net, src, tar, -1):
        net.add_edge(src, tar, codeath=True)
        net.add_edge(tar, src, codeath=True)


def add_merge(net, event, src, tar):
    if net.nodes[src]['type'] == net.nodes[tar]['type']:
        src_events = net.nodes[src]['object_events']
        tar_events = net.nodes[tar]['object_events']
        if src_events[-1] in tar_events[:-1]:
            net.add_edge(src, tar, merge=True)


def add_inheritance(net, event, src, tar):
    if has_init_node(net, src, tar, event):
        death, birth = has_death_and_birth(net, src, tar)
        if death is not None:
            net.add_edge(death, birth, inheritance=True)


# MAIN FUNCTIONS
class Relations(Enum):
    INTERACTS = add_interaction
    DESCENDANTS = add_descendants
    COBIRTH = add_cobirth
    CODEATH = add_codeath
    MERGE = add_merge
    INHERITANCE = add_inheritance


def create_object_centric_graph(log, relations=None):
    if relations is None:
        relations = [Relations.DESCENDANTS]

    return create_multi_graph(log, relations)
