import ocel
from math import isnan
from copy import copy

OBJECTS = 'ocel:objects'
EVENTS = 'ocel:events'
TIMESTAMP = 'ocel:timestamp'


def omap_list_to_set(log):
    for event_id in log['ocel:events']:
        event = log['ocel:events'][event_id]
        event['ocel:omap'] = set(event['ocel:omap'])


def get_activity_names(log):
    e_list = ocel.get_events(log)
    activity_names = set()

    for en, ev in e_list.items():
        activity_names.add(ev['ocel:activity'])

    return list(activity_names)


def get_an_trace(log, eids):
    events = log['ocel:events']
    return [events[e]['ocel:activity'] for e in eids]


def remove_empty_entities(log):
    objs = ocel.get_objects(log)
    events = ocel.get_events(log)
    if "" in objs:
        del objs[""]
    if "" in events:
        del events[""]

    return log


def get_common_attribute_names(log, oids=None, obj_type=None):
    if obj_type is None:
        obj_type = set(ocel.get_object_types(log))
    else:
        obj_type = set(obj_type)

    attribute_dict = {an: 0.0 for an in ocel.get_attribute_names(log)}

    if oids:
        objs = {o: v for o, v in log['ocel:objects'].items()
                if v['ocel:type'] == obj_type and o in oids}
    else:
        objs = {o: v for o, v in log['ocel:objects'].items()
                if v['ocel:type'] == obj_type}

    for o in objs:
        if objs[o]['ocel:type'] in obj_type:
            ovmap = objs[o]['ocel:ovmap']
            for ov in ovmap:
                if ovmap[ov] is not None and (isinstance(ovmap[ov], float)
                   and not isnan(ovmap[ov])):
                    attribute_dict[ov] = attribute_dict[ov] + (1 / len(objs))

    return {an: av for an, av in attribute_dict.items() if av != 0.0}


def get_single_object_events(log, oid):
    events = ocel.get_events(log)
    return {e for e in events if oid in events[e]['ocel:omap']}


def get_multi_object_events(log, oid_list=None):

    if oid_list is None:
        oid_list = ocel.get_objects(log)

    events = ocel.get_events(log)
    obj_event_dict = {oid: list() for oid in oid_list}

    for e_k, e_v in events.items():
        for o in e_v['ocel:omap']:
            if o in obj_event_dict.keys():
                obj_event_dict[o].append(e_k)

    return obj_event_dict


def create_subproblem(log, graph, oids: set, eids: set):
    subgraph = graph.subgraph(oids)
    sublog = copy(log)
    events = log['ocel:events']
    objects = log['ocel:objects']
    sublog['ocel:events'] = {k: events[k] for k in events if k in eids}
    sublog['ocel:objects'] = {k: objects[k] for k in objects if k in oids}

    return sublog, subgraph


def create_sublog(log, oids, eids):
    sublog = copy(log)
    events = log['ocel:events']
    objects = log['ocel:objects']
    sublog['ocel:events'] = {k: events[k] for k in events if k in eids}
    sublog['ocel:objects'] = {k: objects[k] for k in objects if k in oids}

    return sublog


def get_relevant_events(log, graph, oids, until_e):
    events = log['ocel:events']
    to_add = set()
    final_time = events[until_e]['ocel:timestamp']
    for o in oids:
        oe = graph.nodes[o]['object_events']
        to_add |= {e for e in oe if events[e]['ocel:timestamp'] < final_time}

    return to_add


def get_last_event(log):
    return max(log['ocel:events'],
               key=lambda k: log['ocel:events'][k]['ocel:timestamp'])


def get_first_event(log):
    return min(log['ocel:events'],
               key=lambda k: log['ocel:events'][k]['ocel:timestamp'])


def filter_n_events(log, n):
    new_e_dict = {}
    for i, t in enumerate(log['ocel:events'].items()):
        if i >= n:
            break

        new_e_dict[t[0]] = t[1]

    log['ocel:events'] = new_e_dict


def remove_ot(log, ot):
    ot = ot if isinstance(ot, set) else set(ot)
    events = log['ocel:events']
    objects = log['ocel:objects']

    # remove ot from events
    for k, v in events.items():
        v['ocel:omap'] = {x for x in v['ocel:omap']
                          if objects[x]['ocel:type'] not in ot}

    # remove from all objects
    log['ocel:objects'] = {k: v for k, v in objects
                           if v['ocel:type'] not in ot}

# def filter_log_by_subgraph(log, subgraph):
#     new_log = copy(log)
#     events = log['ocel:events']
#     objects = log['ocel:objects']

#     e_add = {}
#     o_add = {}

#     for e in events:
#         if events[e]['ocel:omap'] & oids:
#             e_add[e] = events[e]

#     for o in objects:
#         if o in oids:
#             o_add[o] = objects[o]

#     new_log['ocel:events'] = e_add
#     new_log['ocel:objects'] = o_add

#     return new_log
