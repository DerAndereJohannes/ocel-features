import ocel


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
