# import heapq
from itertools import chain
from collections import Counter
from scipy.stats import norm, uniform, weibull_min, expon
from datetime import time, timedelta, date
# from pm4py.algo.discovery.inductive.variants.im_clean.algorithm \
# import from_dfg as dfg2net
# from pm4py.objects.petri_net.obj import Marking


def init_log(activity_options, object_options):
    att_names = set(chain(*[ot['properties'] for ot in object_options
                            if ot is not None])) \
        | set(chain(*[acn['properties'] for acn in activity_options
                      if acn is not None]))
    otypes = {ot['name'] for ot in object_options}
    log = {'ocel:objects': {},
           'ocel:events': {},
           'ocel:global-log': {'ocel:version': '0.1',
                               'ocel:ordering': 'timestamp',
                               'ocel:attribute-names': att_names,
                               'ocel:object-types': otypes}
           }

    return log


def generate_events(global_options, activity_options, object_options, objects):
    # gather root events and set time variables
    curr_time, time_end, interval = global_options["timeframe"]
    root_events = [values for e in activity_options for key,
                   values in e.items() if values['is_root']]
    # key: oid, value: tuple(current_object_state, list(Marking))
    state_update = {}
    # heapq prio calc on time passed since event timestamp and object priority.
    # infinity if time not there yet.
    events = []
    # (timestamp, {"g_prio": int, "obj from prev": set,
    # "potential next events": set})
    active_events = []
    # traverse through event log timeframe, add root events
    while curr_time < time_end:
        print("adding start events")
        # add new root events
        generate_root_events(events, objects, root_events,
                             state_update, curr_time)

        # add events based on currently in progress events
        progress_active_events(
            events, objects, active_events, state_update, curr_time)

        curr_time = curr_time + interval

    # sort and rename events
    events = {'e{i}': kv[1] for i, kv in enumerate(
        sorted(events.items(), key=lambda event: event[1]['ocel:timestamp']))}
    # return starting events and starting objects
    return events, objects


def generate_new_event(activity, input_objects, curr_time):
    new_event = {}
    new_event['ocel:activity'] = activity['name']
    new_event['ocel:timestamp'] = curr_time + \
        (compute_rvs((uniform, {'loc': 0, 'scale': 1})) * interval)

    # objects
    new_event['ocel:omap'] = input_objects
    # properties
    new_event['ocel:vmap'] = {}

    # generate properties
    if (properties := activity['properties']):
        for prop, value in properties.items():
            new_event['ocel:vmap'][prop] = compute_rvs(value)

    # create new objects
    new_objects = {}
    if (output_obj := activity['output_obj']):
        for ot, freq in output_obj.items():
            if not isinstance(freq, int):
                freq = round(compute_rvs(freq))
            for i in range(freq):
                oid, new_object = generate_new_object(object_options[ot])
                new_objects[oid] = new_object
                new_event['ocel:omap'].add(oid)

    return new_event, new_objects
    events[f'e{len(events)}'] = new_event


def generate_new_object(object_type):
    new_object = {}
    oid = f'{object_type["name"]}{object_type["counter"]}'
    object_type['counter'] = object_type['counter'] + 1
    new_object['ocel:type'] = object_type['name']
    new_object['ocel:ovmap'] = {}

    # add properties
    if object_type['properties']:
        for prop, value in object_type['properties'].items():
            new_object['ocel:ovmap'][prop] = compute_rvs(value)

    return oid, new_object


def generate_root_events(events, objects, root_events,
                         state_update, curr_time):
    root_events = []
    for event in root_events:
        if in_active_time(curr_time, event['active_time']):
            frequency = int(compute_rvs(event['interval_quantity']))
            new_event = (curr_time, {'prev_obj': set(),
                         'event_choice': {event['name']}})
            for i in range(frequency):
                root_events.append(new_event)

    progress_active_events(events, objects, root_events,
                           state_update, curr_time)


def progress_active_events(events, objects, active_events,
                           state_update, curr_time, interval):
    # events: official events
    # objects: official objects
    # active_events: events to use in format =
    # (timestamp, (event obj, input_obj: set))
    # state_update: (og obj with edits to values, locked_to_objects)
    # curr_time: current time

    # setup. gather expired events
    expired_events = []
    for e in active_events:
        if e[0] <= curr_time:
            expired_events.append(e[1])

    active_events = active_events[len(expired_events):]

    # 1. execute transition for each event and object (create new objects too)
    for e, input_obj in expired_events:
        # execute the event
        activity = activity_options[e['name']]
        new_event, new_objects = generate_new_event(
            activity, input_obj, curr_time)

        # update existing objects
        for oid in input_obj:
            state_update[oid][1] = e['name']
            state_update[oid][0] = update_object_values(
                state_update[oid][0], e)

        # add new event to events
        events[f'e{len(events)}'] = new_event

        # add new objects to objects and state
        for oid, o in new_objects.items():
            state_update[oid] = [o, e['name'], set()]

        object_output = new_event['ocel:objects']

        # 2. check which new transitions are enabled
        while (possible_transitions := get_possible_successors(object_output,
                                                               state_update)):
            pass


def update_object_values(obj, event):
    pass


def get_possible_successors():
    # if event is a leaf, return none
    pass


def get_next_event(finished_event):
    pass


def get_object_property(objects, state_dict, obj, prop):
    if (key_str := f'{obj}:{prop}') in state_dict:
        return state_dict[key_str]
    elif prop in objects[obj]['ocel:ovmap'][prop]:
        return objects[obj]['ocel:ovmap'][prop]

    return None


def generate_ocel(global_options, activities, object_types):
    # check if all required global options are set. If not, set default.
    validate_generation_parameters(global_options, activities, object_types)
    # setup log and dictionary objects
    log = init_log(global_options)
    objects = create_starting_objects(global_options)
    events = generate_events(
        global_options, activities, object_types, objects)

    log['ocel:objects'] = objects
    log['ocel:events'] = events

    return log


def validate_generation_parameters(global_options, activities, objects):
    required = ['timeframe', ]
    possible = ['starting_objects', 'constraints']
    required_activity = ['name', 'properties', 'df']

    # exit if all required parameters are not included
    if (not_contained := [req for req in required
                          if req not in global_options]):
        print(f'Required parameters are missing: {not_contained}')
        return

    # prepare object type counters
    global_options['ot_counters'] = {
        ot: 0 for ot in global_options['object_types']}


def create_starting_objects(global_options):
    starting_object_options = global_options['starting_objects']
    objects = {}
    for ot, obj_props in starting_object_options.items():
        if isinstance(obj_props, int):
            # simple numbered names
            names = [f'{ot}:{i}' for i in range(obj_props)]
        else:
            # names defined by user
            names = [f'{ot}:{name}' for name in obj_props]

        for name in names:
            objects[name] = create_object(ot)
        # increment ot_counter
        global_options['ot_counters'][ot] = len(names)

    return objects


def create_object(ot_dict):
    new_object = {"ocel:type": ot_dict['name'],
                  "ocel:ovmap": {}
                  }

    # return if object has no properties
    if ot_dict['properties'] is None:
        return new_object

    # generate new properties
    for prop, rvs_tup in ot_dict['properties'].items():
        new_object['ocel:ovmap'][prop] = compute_rvs(rvs_tup)

    return new_object


def in_active_time(curr_time, active_tuple):
    if not active_tuple:
        return False

    start_t, end_t, days_of_week = active_tuple

    # check if the curr time is in correct day of week
    dow = curr_time.weekday() in days_of_week

    # check if curr time is in correct time
    t = start_t < curr_time.time() < end_t

    return dow and t


def compute_rvs(prob_tuple):
    f, params = prob_tuple

    return f.rvs(**params)


if __name__ == '__main__':
    global_options = {"timeframe": (date(2022, 5, 1),
                                    date(2022, 5, 31),
                                    timedelta(minutes=5)),
                      "starting_objects": {"employee": {"Mike", "Frank",
                                                        "Cyrille", "Justin",
                                                        "Louis", "Kate",
                                                        "Julia", "Alice",
                                                        "June", "Carly"},
                                           "system": {'SYS'}},
                      "constraints": [('employee', 'energy',
                                       lambda x: x >= 0)]}
    activity_options = [{}]
    object_options = [{}]

    generate_ocel(global_options, activity_options, object_options)


employee = {
    "name": "employee",
            "df": None,
            "properties": {
                "energy": (norm, {"loc": 100, "scale": 3})
            }
}


system = {
    "name": "system",
            "df": None,
            "properties": None
}


order = {
    "name": "order",
    "df": Counter([('create order', 'check availability'),
                   ('create order', 'accept order'),
                   ('check availability', 'accept order'),
                   ('accept order', 'check availability'),
                   ('check availability', 'check availability'),
                   ('accept order', 'pick item'),
                   ('check availability', 'pick item'),
                   ('pick item', 'pick item'),
                   ('check availability', 'check availability'),
                   ('accept order', 'send invoice'),
                   ('send invoice', 'receive payment'),
                   ('accept order', 'pick item')]),
    "properties": {
        "Priority": (uniform, {"loc": 0, "scale": 10})
    }
}

item = {
    "name": "item",
    "df": Counter([('check availability', 'pick item'),
                   ('pick item', 'pack items')]),
    "properties": {
            "weight(g)": (norm, {"loc": 160, "scale": 5}),
            "cost($)": (norm, {"loc": 500, "scale": 100})
    }
}

package = {
    "name": "package",
    "df": Counter([('pack items', 'store package'),
                   ('store package', 'start route'),
                   ('start route', 'load package'),
                   ('load package', 'deliver package'),
                   ('load package', 'failed delivery'),
                   ('failed delivery', 'unload package'),
                   ('unload package', 'store package'),
                   ('load package', 'transfer package'),
                   ('transfer package', 'unload package')]),
    "properties": None
}

route = {
    "name": "route",
    "df": Counter([('start route', 'load package'),
                   ('load package', 'load package'),
                   ('load package', 'transfer package'),
                   ('load package', 'deliver package'),
                   ('load package', 'failed delivery'),
                   ('transfer package', 'transfer package'),
                   ('deliver package', 'deliver package'),
                   ('failed delivery', 'failed delivery'),
                   ('deliver package', 'transfer package'),
                   ('failed delivery', 'deliver package'),
                   ('deliver package', 'failed delivery'),
                   ('failed delivery', 'unload package'),
                   ('transfer package', 'unload package'),
                   ('unload package', 'end route'),
                   ('deliver package', 'deliver package'),
                   ('deliver package', 'end route')
                   ('unload package', 'unload package')]),
    "properties": None
}

create_order = {
    "is_root": True,
    "is_leaf": False,
    "interval_quantity": (expon, {"loc": 1, "scale": 0.5}),
    "name": "create order",
    "active_time": None,
    "sojourn_dist": (weibull_min, {"c": 2, "loc": 100, "scale": 100}),
    "input_obj": {"system": 1},
    "output_obj": {"item": (expon, {"scale": 0.5}), "order": 1},
    "input_events": None,
    "output_events": {"accept order", "check availability"},
    "properties": None
}

check_avail = {
    "name": "check availability",
    "active_time": (time(9), time(17), list(range(5))),
    "sojourn_dist": (weibull_min, {"c": 2, "loc": 10, "scale": 20}),
    "input_obj": {"employee": 1, "item": 1, "order": 1},
    "output_obj": None,
    "input_events": {"create order", "check availability", "accept order"},
    "output_events": {"pick item"},
    "properties": None
}

pick_item = {
    "name": "pick item",
    "active_time": (time(9), time(17), list(range(5))),
    "sojourn_dist": (weibull_min, {"c": 2, "loc": 0, "scale": 100}),
    "input_obj": {"employee": 1, "item": 1, "order": 1},
    "output_obj": None,
    "input_events": {"check availability"},
    "output_events": {},
    "properties": None
}
