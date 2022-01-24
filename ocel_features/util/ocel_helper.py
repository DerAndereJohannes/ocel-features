import ocel
from math import isnan


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


def remove_empty_entities(log):
    objs = ocel.get_objects(log)
    events = ocel.get_events(log)
    if "" in objs:
        del objs[""]
    if "" in events:
        del events[""]

    return log


def get_common_attribute_names(log, obj_type=None):
    if obj_type is None:
        obj_type = set(ocel.get_object_types(log))
    else:
        obj_type = set(obj_type)

    attribute_dict = {an: 0.0 for an in ocel.get_attribute_names(log)}
    all_objs = ocel.get_objects(log)
    objs = {o: v for o, v in all_objs.items()
            if all_objs[o]['ocel:type'] in obj_type}

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
