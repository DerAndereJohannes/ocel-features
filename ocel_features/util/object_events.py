import ocel


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
