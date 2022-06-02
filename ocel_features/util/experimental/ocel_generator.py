import sys
import heapq
import random
import networkx as nx
import numpy as np
from itertools import chain
from collections import Counter
from scipy.stats import norm, uniform, weibull_min, expon, bernoulli
# from random import randint
from datetime import time, timedelta, date, datetime
# from pm4py.algo.discovery.inductive.variants.im_clean.algorithm \
# import from_dfg as dfg2net
# from pm4py.objects.petri_net.obj import Marking


def get_obj_activity_options(global_options):
    object_options = global_options['object_types']
    process = next(iter(global_options['processes'].values()))  # TODO
    activity_options = {n: process.nodes[n] for n in process.nodes()}
    return object_options, activity_options


def init_log(global_options):
    object_options, activity_options = get_obj_activity_options(global_options)

    att_names = set(chain(*[ot['properties'] for ot in object_options.values()
                            if ot['properties'] is not None])) \
        | set(chain(*[acn['properties'] for acn in activity_options.values()
                      if acn['properties'] is not None]))
    otypes = {ot['name'] for ot in object_options.values()}
    log = {'ocel:objects': {},
           'ocel:events': {},
           'ocel:global-log': {'ocel:version': '0.1',
                               'ocel:ordering': 'timestamp',
                               'ocel:attribute-names': att_names,
                               'ocel:object-types': otypes}
           }

    return log


def generate_events(global_options):
    object_options, activity_options = get_obj_activity_options(global_options)
    # gather root events and set time variables
    curr_time, time_end, interval = global_options["timeframe"]
    root_events = [values for values in activity_options.values()
                   if values['interval_quantity']]

    # traverse through event log timeframe, add root events
    while curr_time < time_end:
        print("adding start events")
        # add new root events
        generate_root_events(global_options, root_events, curr_time)
        # print_debug(global_options)
        # sys.exit()
        # add events based on currently in progress events
        progress_active_events(global_options, curr_time)

        curr_time = curr_time + interval

    # sort and rename events
    events = global_options['log']['ocel:events']
    events = {'e{i}': kv[1] for i, kv in enumerate(
        sorted(events.items(), key=lambda event: event[1]['ocel:timestamp']))}

    global_options['log']['ocel:events'] = events


def generate_new_event(activity, input_objects, curr_time, interval,
                       object_options):
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
    # events[f'e{len(events)}'] = new_event


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


def generate_root_events(go, root_events, curr_time):
    for event in root_events:
        if in_active_time(curr_time, event['active_time']):
            process = go['processes'][event['process_name']]
            successors = process.successors(event['name'])
            frequency = round(compute_rvs(event['interval_quantity']))

            new_event = create_root_active_event(event, successors, curr_time)
            for i in range(frequency):
                heapq.heappush(go['active_events'], new_event)


def create_root_active_event(event, successors, curr_time):
    return (curr_time, (event['process_name'],  # graph used    [1][0]
                        event['name'],          # activity name [1][1]
                        frozenset({}),          # objects       [1][2]
                        "service"))             # type of event [1][3]


def progress_active_events(go, curr_time):
    # events: official events
    # objects: official objects
    # active_events: events to use in format =
    # (timestamp, (event obj, obj_on_this_step, was_event_readded?))
    # state_update: (og obj with edits to values, locked_to_objects)
    # curr_time: current time

    # setup. gather expired events
    expired_events = []
    for e in go['active_events']:
        if e[0] <= curr_time:
            expired_events.append(e)
        else:
            break

    # setup global events
    global_events = []
    for e in go['active_events'][::-1]:
        if e[1][1] == 'global objects':
            global_events.append(e)
        else:
            break

    go['active_events'] = go['active_events'][len(expired_events):]
    events = []
    readd_events = []

    # Create global representation for the expired events
    expired_view = generate_expired_state_view(go,
                                               expired_events, global_events)

    # 1. execute transition for each event and object (create new objects too)
    for timestamp, active_event in expired_events:
        process = go['processes'][active_event[0]]
        activity = active_event[1]
        objects = active_event[2]
        event_type = active_event[3]

        src_an_obj = process.nodes[activity]
        successors = process.successors(activity)
        remaining_objects = set()

        # if the current time does not suit the active time of the activity
        if not in_active_time(curr_time, src_an_obj['active_time']):
            readd_events.append((timestamp, active_event))
            continue

        # passthrough wait event types and convert service types to wait types
        # output: active_events = (timestamp, (process, act, obj, ev_type))
        if event_type == 'wait':
            active_events, readd_new = [(timestamp, active_event), ], []
        elif event_type == 'service':
            # return list of successors and remainder events
            active_events, readd_new = generate_successors(process,
                                                           timestamp,
                                                           active_event,
                                                           successors)
            for e in readd_new:
                heapq.heappush(readd_events, e)

        available_objects = generate_expired_state_view(go, active_events)

        # activate all wait types and add to events. update all active events
        # to service types
        for _, event in active_events:
            process = go['processes'][event[0]]
            activity, objects, event_type = event[1:]

            input_objects = gather_objects(go, event, expired_view)

            if input_objects:
                transition = generate_new_event(activity, input_objects,
                                                curr_time)
                apply_new_transition(go, transition)
            else:
                active_event = (active_event[:])
                heapq.heappush(readd_events, (timestamp, active_event))
                continue

        # while (next_transition := get_next_transition(go, process,
        #                                               active_event,
        #                                               expired_view,
        #                                               curr_time)):
        #     new_event, new_objects = next_transition[0]  # to add to events
        #     next_active = next_transition[1]
        #     remaining_objects = next_transition[2]  # to add to readd events

        #     # add new event to events
        #     new_event_id = f'e{len(events)}'
        #     events[new_event_id] = new_event

        #     # update existing objects
        #     for oid in input_obj:
        #         update_object_values(state_update, src_an_obj, oid, new_event)

        #     # add new objects to objects and state
        #     for oid, o in new_objects.items():
        #         state_update[oid] = [o, new_event['ocel:name'], dict()]

        #     # update active events
        #     heapq.heappush(active_events, next_active)

        # if remaining_objects:
        #     readd_events.append((curr_time, (src_an, remaining_objects)))

        # # 3. readd any events that did not manage to process all objects
        # go['active_events'] = readd_events + go['active_events']
        # go['log']['ocel:events'].extend(events)


def generate_successors(process, timestamp, active_event, successors):
    # loop
    random_successor = np.random.permutation(successors)[0]
    new_events = []
    readd_events = []
    flow_relation = process.nodes[random_successor]['flow_relation']
    flow = None
    # Check if there is a special flow
    if flow_relation:
        for relation, relation_connections in flow_relation.items():
            for combos in relation_connections:
                if random_successor in combos:
                    flow = (relation, combos)
                    break

    # create new successors depending on flow
    if flow:
        relation, activities = flow
        if relation == 'and':
            for act in activities:
                new_active_event = list(active_event)
                new_active_event[1] = act
                new_active_event[3] = 'wait'
                wait_time = process.nodes[act]['wait_time']
                if wait_time:
                    new_timestamp = timestamp + int(compute_rvs(wait_time))
                    if new_timestamp < timestamp: # CONTINUTE HERE
                    readd_events.append((new_timestamp, new_active_event))
                else:
                    new_events.append((timestamp, new_active_event))

    else:
        # if no special flow, simply pass on
        new_active_event = list(active_event)
        new_active_event[1] = random_successor
        new_active_event[3] = 'wait'
        wait_time = process.nodes[act]['wait_time']
        if wait_time:
            new_timestamp = timestamp + int(compute_rvs(wait_time))
            readd_events.append((new_timestamp, new_active_event))
        else:
            new_events.append((timestamp, new_active_event))

    return new_events, readd_events


def gather_objects(act_obj, input_obj, expired_view):
    pass


def obj_marking_key(go, objs):
    objects = go['log']['ocel:objects']
    if isinstance(objs, tuple):
        return frozenset({objects[o]['ocel:type'] for o in objs})
    else:
        return objects[objs]['ocel:type']


def generate_expired_state_view(go, expired_events, global_events):
    # generate dictionary with all active activity names
    expired_global_view = {}

    # get all global objects to add to all activity names
    global_objects = set()
    for global_event in global_events:
        global_objects |= global_event[2]

    for expired_event in expired_events:
        process, activity_name, remaining_objects, _ = expired_event
        if activity_name not in expired_global_view:
            expired_global_view[activity_name] = global_objects

        # if the activity is not locked, add all the objects to the pool
        lock = go['processes'][process].nodes[activity_name]['object_relation']
        if not lock:
            expired_global_view[activity_name] |= remaining_objects

    return expired_global_view


def update_object_values(obj_states, src_act, oid, new_event):
    # update object values based on changing values
    curr_o = obj_states[oid][0]

    # changes here

    # update the state
    obj_states[oid][0] = curr_o


def get_next_transition(go, process, active_event, expired_view, curr_time):
    curr_act = active_event[1]
    curr_objs = active_event[2]

    successors = process.successors(curr_act)

    # 1. if event is a leaf, return none
    if not successors:
        return None, None, None

    act, input_objs = pick_successor(go, successors, active_event,
                                     expired_view, process, curr_time)
    next_act = process.nodes[act]
    expiry_date = curr_time + int(compute_rvs(next_act['service_time']))
    transition = generate_new_event(next_act, input_objs, curr_time)
    next_active = (expiry_date, (act, transition['ocel:omap']))
    remaining_objects = curr_objs - input_objs

    return transition, next_active, remaining_objects


def pick_successor(go, successors, active_event, expired_view,
                   process, curr_time):
    curr_act = active_event[1]
    curr_objs = active_event[2]
    allowed_successors = []

    for s in successors:
        activity = process.nodes[s]
        # active time
        if not in_active_time(curr_time, activity['active_time']):
            continue
        # check object possibility
        locked_relation = activity['locked_relation']
        if locked_relation == 'locked':
            pass
        else:
            pass  # TODO

        allowed_successors.append(s)

    possible_successors = []
    # remove any successors that cant handle the flow relations
    for s in allowed_successors:
        activity = process.nodes[s]
        flows = activity['flow_relation']

        if not flows:
            possible_successors.append(s)
            continue

        # and relations
        ands = activity['flow_relation']['and']
        flow_check = {}
        for flow in ands:
            flow_check.add(all(a in allowed_successors for a in flow))

        if True in flow_check:
            possible_successors.append(s)
            continue

    if not possible_successors:
        return None, None
    # select successor
    chosen_act = random.choice(possible_successors)

    # populate event
    chosen_obj = select_input_objects(go, chosen_act)

    return chosen_act, chosen_obj


def select_input_objects(go, act):
    pass


def get_object_property(objects, state_dict, obj, prop):
    if (key_str := f'{obj}:{prop}') in state_dict:
        return state_dict[key_str]
    elif prop in objects[obj]['ocel:ovmap'][prop]:
        return objects[obj]['ocel:ovmap'][prop]

    return None


def generate_ocel(global_options):
    # check if all required global options are set. If not, set default.
    validate_generation_parameters(global_options)
    # setup state holders
    global_options['object_state'] = {}
    global_options['ot_dir'] = {ot: set()
                                for ot in global_options['object_types']}
    print(global_options['ot_dir'])
    global_options['active_events'] = []

    # setup log and dictionary objects
    log = init_log(global_options)
    log['ocel:objects'] = create_starting_objects(global_options)

    global_options['log'] = log
    generate_events(global_options)

    return log


def validate_generation_parameters(global_options):
    required = ['timeframe', 'time_unit', 'processes', 'object_types']
    # possible = ['starting_objects', 'global_constraints']
    # required_activity = ['name', 'properties']
    # required_object = ['type', 'properties']

    # exit if all required parameters are not included
    if (not_contained := [req for req in required
                          if req not in global_options]):
        print(f'Required parameters are missing: {not_contained}')
        sys.exit()


def create_global_event(objs):
    return (datetime.max, (None, 'global objects', frozenset(objs), "service"))


def create_starting_objects(global_options):
    object_options = global_options['object_types']
    starting_object_options = global_options['starting_objects']
    objects = {}
    for ot, obj_props in starting_object_options.items():
        if isinstance(obj_props, int):
            # simple numbered names
            names = [f'{ot[:4]}:{i}' for i in range(obj_props)]
        else:
            # names defined by user
            names = [f'{ot[:4]}:{name}' for name in obj_props]

        for name in names:
            objects[name] = create_starting_object(object_options[ot])
            global_options['object_state'][name] = {'state': objects[name],
                                                    'locked_objects': {},
                                                    'active_locks': set(),
                                                    'active_events': set()}
            global_options['ot_dir'][objects[name]['ocel:type']].add(name)
        # increment ot_counter
        object_options[ot]['counter'] = len(names)
    active_event = create_global_event(objects.keys())
    global_options['active_events'].append(active_event)

    for oid_values in global_options['object_state'].values():
        oid_values['active_events'].add(active_event)

    return objects


def create_starting_object(ot_dict):
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
        return True

    start_t, end_t, days_of_week = active_tuple

    # check if the curr time is in correct day of week
    dow = curr_time.weekday() in days_of_week

    # check if curr time is in correct time
    t = start_t < curr_time.time() < end_t

    return dow and t


def compute_rvs(prob_tuple):
    f, params = prob_tuple

    return f.rvs(**params)


employee = {
    "name": "employee",
    # "df": None,
    "constraints": [('energy', lambda x: x >= 0)],
    "properties": {
        "energy": (norm, {"loc": 100, "scale": 3})
    }

}


system = {
    "name": "system",
    # "df": None,
    "properties": None
}


order = {
    "name": "order",
    # "df": Counter([('create order', 'check availability'),
    #                ('create order', 'accept order'),
    #                ('check availability', 'accept order'),
    #                ('accept order', 'check availability'),
    #                ('check availability', 'check availability'),
    #                ('accept order', 'pick item'),
    #                ('check availability', 'pick item'),
    #                ('pick item', 'pick item'),
    #                ('check availability', 'check availability'),
    #                ('accept order', 'send invoice'),
    #                ('send invoice', 'receive payment'),
    #                ('accept order', 'pick item')]),
    "properties": {
        "Priority": (uniform, {"loc": 0, "scale": 10})
    }
}

item = {
    "name": "item",
    # "df": Counter([('check availability', 'pick item'),
    #                ('pick item', 'pack items')]),
    "properties": {
            "weight(g)": (norm, {"loc": 160, "scale": 5}),
            "cost($)": (norm, {"loc": 500, "scale": 100})
    }
}

package = {
    "name": "package",
    # "df": Counter([('pack items', 'store package'),
    #                ('store package', 'start route'),
    #                ('start route', 'load package'),
    #                ('load package', 'deliver package'),
    #                ('load package', 'failed delivery'),
    #                ('failed delivery', 'unload package'),
    #                ('unload package', 'store package'),
    #                ('load package', 'transfer package'),
    #                ('transfer package', 'unload package')]),
    "properties": {"_completed": False}
}

route = {
    "name": "route",
    # "df": Counter([('start route', 'load package'),
    #                ('load package', 'load package'),
    #                ('load package', 'transfer package'),
    #                ('load package', 'deliver package'),
    #                ('load package', 'failed delivery'),
    #                ('transfer package', 'transfer package'),
    #                ('deliver package', 'deliver package'),
    #                ('failed delivery', 'failed delivery'),
    #                ('deliver package', 'transfer package'),
    #                ('failed delivery', 'deliver package'),
    #                ('deliver package', 'failed delivery'),
    #                ('failed delivery', 'unload package'),
    #                ('transfer package', 'unload package'),
    #                ('unload package', 'end route'),
    #                ('deliver package', 'deliver package'),
    #                ('deliver package', 'end route'),
    #                ('unload package', 'unload package')]),
    "properties": None
}


def create_example_process():
    # create process
    process_df = [('setup order', 'create order'),
                  ('create order', 'accept order'),
                  ('accept order', 'receive payment'),
                  ('accept order', 'check availability'),
                  ('check availability', 'pick item'),
                  ('pick item', 'check availability'),
                  ('pick item', 'pack items'),
                  ('pack items', 'store package'),
                  ('store package', 'start route'),
                  ('setup route', 'start route'),
                  ('start route', 'load package'),
                  ('load package', 'fail delivery'),
                  ('load package', 'deliver package'),
                  ('fail delivery', 'unload package'),
                  ('unload package', 'store package'),
                  ('unload package', 'end route'),
                  ('deliver package', 'end route')]
    process = nx.DiGraph(process_df)
    process.add_node('setup order', **setup_order)
    process.add_node('create order', **create_order)
    process.add_node('accept order', **accept_order)
    process.add_node('check availability', **check_availability)
    process.add_node('receive payment', **receive_payment)
    process.add_node('pick item', **pick_item)
    process.add_node('pack items', **pack_items)
    process.add_node('store package', **store_package)
    process.add_node('setup route', **setup_route)
    process.add_node('start route', **start_route)
    process.add_node('load package', **load_package)
    process.add_node('fail delivery', **fail_delivery)
    process.add_node('deliver package', **deliver_package)
    process.add_node('unload package', **unload_package)
    process.add_node('end route', **end_route)

    return process


setup_order = {
    "process_name": "delivery",
    "name": "setup order",
    "interval_quantity": (expon, {"loc": 1, "scale": 0.5}),
    "active_time": None,
    "wait_time": None,
    "service_time": (uniform, {"loc": 1, "scale": 0}),
    "input_obj": None,
    "output_obj": None,
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": None,
    "properties": None
}

create_order = {
    "process_name": "delivery",
    "name": "create order",
    "interval_quantity": None,
    "active_time": None,
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 100, "scale": 100}),
    "input_obj": {"system": 1},
    "output_obj": {"item": (expon, {"scale": 0.5}), "order": 1},
    "flow_relation": {'and': [('receive payment', 'check  availability')]},
    "lock_relation": {('lock', ('order', 'item'))},
    "object_relation": None,
    "properties": None
}

accept_order = {
    "process_name": "delivery",
    "name": "accept order",
    "interval_quantity": None,
    "active_time": None,
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 100, "scale": 100}),
    "input_obj": {frozenset({"order", "item"}): "all"},
    "output_obj": {"item": (expon, {"scale": 0.5}), "order": 1},
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": "locked",
    "properties": None
}

check_availability = {
    "process_name": "delivery",
    "name": "check availability",
    "interval_quantity": None,
    "active_time": (time(9), time(17), list(range(5))),
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 10, "scale": 20}),
    "input_obj": {"employee": 1, "item": 1, "order": 1},
    "output_obj": None,
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": "locked",
    "properties": None
}

receive_payment = {
    "process_name": "delivery",
    "name": "receive payment",
    "interval_quantity": None,
    "active_time": None,
    "wait_time": (weibull_min, {"c": 2, "loc": 10, "scale": 20}),
    "service_time": None,
    "input_obj": {"system": 1, "item": "Any", "order": 1},
    "output_obj": None,
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": "locked",
    "properties": {'payment type': {'credit card', 'debit card',
                                    'bank transfer'}
                   }
}

pick_item = {
    "process_name": "delivery",
    "name": "pick item",
    "interval_quantity": None,
    "active_time": (time(9), time(17), list(range(5))),
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 10, "scale": 100}),
    "input_obj": {"employee": 1, "item": 1, "order": 1},
    "output_obj": None,
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": "locked",
    "properties": None
}

pack_items = {
    "process_name": "delivery",
    "name": "pack items",
    "interval_quantity": None,
    "active_time": (time(9), time(17), list(range(5))),
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 10, "scale": 100}),
    "input_obj": {"employee": 1, "item": 4},
    "output_obj": {"package": 1},
    "flow_relation": None,
    "lock_relation": {('lock', ('item', 'package'))},
    "object_relation": None,
    "properties": None
}

store_package = {
    "process_name": "delivery",
    "name": "store package",
    "interval_quantity": None,
    "active_time": None,
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 10, "scale": 100}),
    "input_obj": {"employee": 1, "item": 1},
    "output_obj": None,
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": "locked",
    "properties": None
}

setup_route = {
    "process_name": "delivery",
    "name": "setup route",
    "interval_quantity": (bernoulli, {"p": 0.25}),
    "active_time": (time(9), time(17), list(range(5))),
    "wait_time": None,
    "service_time": (uniform, {"loc": 1, "scale": 0}),
    "input_obj": None,
    "output_obj": None,
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": None,
    "properties": None
}

start_route = {
    "process_name": "delivery",
    "name": "start route",
    "interval_quantity": None,
    "active_time": None,
    "wait_time": None,
    "service_time": (uniform, {"loc": 10, "scale": 100}),
    "input_obj": {"employee": 1, "package": "Any", "route": 1},
    "output_obj": {"route": 1},
    "flow_relation": None,
    "lock_relation": {('lock', ('package', 'route'))},
    "object_relation": None,
    "properties": None
}

load_package = {
    "process_name": "delivery",
    "name": "load package",
    "interval_quantity": None,
    "active_time": None,
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 5, "scale": 10}),
    "input_obj": {"package": "All", "route": 1},
    "output_obj": None,
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": "locked",
    "properties": None
}

fail_delivery = {
    "process_name": "delivery",
    "name": "fail delivery",
    "interval_quantity": None,
    "active_time": None,
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 20, "scale": 50}),
    "input_obj": {"package": "Any", "route": 1},
    "output_obj": None,
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": "locked",
    "properties": {"_completed": True}
}

deliver_package = {
    "process_name": "delivery",
    "name": "deliver package",
    "interval_quantity": None,
    "active_time": None,
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 20, "scale": 50}),
    "input_obj": {"package": "Any", "route": 1},
    "output_obj": None,
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": "locked",
    "properties": {"_completed": True}
}

unload_package = {
    "process_name": "delivery",
    "name": "unload package",
    "interval_quantity": None,
    "active_time": None,
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 2, "scale": 3}),
    "input_obj": {"package": "All", "route": 1},
    "output_obj": None,
    "flow_relation": {"and": [('store package', 'end route')]},
    "lock_relation": None,
    "object_relation": "locked",
    "constraints": [('package', '_completed', lambda x: x is True)],
    "properties": None
}

end_route = {
    "process_name": "delivery",
    "name": "end route",
    "interval_quantity": None,
    "active_time": None,
    "wait_time": None,
    "service_time": (weibull_min, {"c": 2, "loc": 0, "scale": 1}),
    "input_obj": {"package": "All", "route": 1},
    "output_obj": None,
    "flow_relation": None,
    "lock_relation": None,
    "object_relation": "locked",
    "constraints": [('package', '_completed', lambda x: x is True)],
    "properties": None
}


def print_debug(global_options):
    pprint(global_options)


if __name__ == '__main__':
    from pprint import pprint
    process = create_example_process()
    global_options = {"timeframe": (datetime(2022, 5, 1),
                                    datetime(2022, 5, 31),
                                    timedelta(minutes=5)),
                      "time_unit": "minutes",
                      "starting_objects": {"employee": {"Mike", "Frank",
                                                        "Cyrille", "Justin",
                                                        "Louis", "Kate",
                                                        "Julia", "Alice",
                                                        "June", "Carly"},
                                           "system": {'SYS'}},
                      "processes": {"delivery": process},
                      "object_types": {"employee": employee,
                                       "system": system,
                                       "order": order,
                                       "item": item,
                                       "package": package,
                                       "route": route}
                      }

    log = generate_ocel(global_options)
    pprint(log)
