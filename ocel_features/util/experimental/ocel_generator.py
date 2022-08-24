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

    print("adding start events")
    curr_interval = 0
    # traverse through event log timeframe, add root events
    while curr_time < time_end:
        print(f"executing interval: {curr_interval} -> {curr_time}")
        curr_interval = curr_interval + 1
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
                freq = max(1, round(compute_rvs(freq)))
            for i in range(freq):
                oid, new_object = generate_new_object(object_options[ot])
                new_objects[oid] = new_object
                new_event['ocel:omap'].add(oid)

    return new_event, new_objects
    # events[f'e{len(events)}'] = new_event


def generate_new_object(object_type):
    new_object = {}
    oid = f'{object_type["name"]}0'
    if 'counter' in object_type:
        oid = f'{object_type["name"]}{object_type["counter"]}'
        object_type['counter'] = object_type['counter'] + 1
    else:
        object_type['counter'] = 1
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
    print(f'curr event dist: {Counter([x[1][1] for x in go["active_events"]])}')
    go['active_events'].sort()
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

    go['active_events'] = go['active_events'][len(expired_events):] # we add back later if needed
    events = []
    readd_events = []

    # Create global representation for the expired events
    # expired_view = generate_expired_state_view(go,
    #                                            expired_events, global_events)

    # 1. execute transition for each event and object (create new objects too)
    for timestamp, active_event in expired_events:
        print(f"Progressing: {active_event} at {timestamp}")
        process = go['processes'][active_event[0]]
        activity = active_event[1]
        objects = active_event[2]
        event_type = active_event[3]

        src_an_obj = process.nodes[activity]
        successors = list(process.successors(activity))

        if not successors:
            continue

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

        expired_view = generate_expired_state_view(go, active_events, global_events)

        # activate all wait types and add to events. update all active events
        # to service types
        for _, event in active_events:
            print(f"activating: {event}")
            process = go['processes'][event[0]]
            activity, objects, event_type = event[1:]
            # get all possible event combinations

            # pick event

            # pick object


            input_objects, readd_obj = gather_objects(process.nodes[activity], objects, expired_view, go)
            pprint(input_objects)
            # pprint(go["ot_dir"])
            # validate that all input requirements are fulfilled
            val = Counter([go["object_state"][x]["state"]["ocel:type"] for x in input_objects])
            if input_objects:
                # create new event
                new_event, new_objs = generate_new_event(process.nodes[activity], input_objects,
                                                curr_time, go["timeframe"][2], go["object_types"])

                # update object state
                if (lr := process.nodes[activity]["lock_relation"]):
                    for action, items in lr:
                        rel1 = [(k, v) for k, v in new_objs.items() if v["ocel:type"] == items[0]]
                        rel2 = [(k, v) for k, v in new_objs.items() if v["ocel:type"] == items[1]]
                        for r1 in rel1:
                            for r2 in rel2:
                                update_object_state(go, r1, r2, items)
                                update_object_state(go, r2, r1, items)

                #

                go["log"]["ocel:events"][f'e{len(go["log"]["ocel:events"])}'] = new_event
                go["log"]["ocel:objects"].update(new_objs)
                print(new_event, new_objs)
                # calculate new service time and add to active events
                if process.nodes[activity]['service_time']:
                    service_time = new_event["ocel:timestamp"] + timedelta(minutes=compute_rvs(process.nodes[activity]['service_time']))
                else:
                    service_time = new_event["ocel:timestamp"]

                heapq.heappush(go["active_events"], (service_time,(event[0], event[1], input_objects, "service")))
            else:
                heapq.heappush(readd_events, (timestamp, tuple(active_events)))

            if readd_obj:
                event[2] = readd_obj
                heapq.heappush(readd_events, (timestamp, tuple(active_event)))
            # if active_event[1] == 'create order' and active_event[3] == 'service':
                # if input() == "1":
                #     pprint(go["log"]["ocel:events"])

    for re in readd_events:
        heapq.heappush(go["active_events"], re)
    # pprint(go["log"])
    # pprint(go["active_events"])
    # pprint(go["object_state"])
    # input()


def update_object_state(go, o1, o2, lock):
    if o1[0] in go["object_state"]:
        go['object_state'][o1[0]]['active_locks'].add(lock)

        if o2[1]["ocel:type"] in go["object_state"][o1[0]]['locked_objects']:
            go["object_state"][o1[0]]['locked_objects'][o2[1]["ocel:type"]].add(o1[0])
        else:
            go["object_state"][o1[0]]['locked_objects'][o2[1]["ocel:type"]] = {o1[0]}

    else:
        go['object_state'][o1[0]] = {'state': o2[1],
                                     'locked_objects': {o2[1]["ocel:type"]: {o2[0]}},
                                     'lock_quantity': {o2[1]["ocel:type"]: 1},
                                     'active_locks': {lock}}


def generate_successors(process, timestamp, active_event, successors):
    # loop
    successor_list = list(successors)
    random_successor = np.random.choice(successor_list)
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
                    new_timestamp = timestamp + timedelta(minutes=compute_rvs(wait_time))
                    new_event = (new_timestamp, new_active_event)
                    if new_timestamp > timestamp:
                        readd_events.append(new_event)
                    else:
                        new_events.append(new_event)
                else:
                    new_events.append((timestamp, new_active_event))

    else:
        # if no special flow, simply pass on
        new_active_event = list(active_event)
        new_active_event[1] = random_successor
        new_active_event[3] = 'wait'
        wait_time = process.nodes[random_successor]['wait_time']
        if wait_time:
            new_timestamp = timestamp + timedelta(minutes=compute_rvs(wait_time))
            readd_events.append((new_timestamp, new_active_event))
        else:
            new_events.append((timestamp, new_active_event))

    return new_events, readd_events


def get_obj_quantity_view(go, activity_dict, input_objects):
    existing_objs = {}

    for ot, quantity in activity_dict["input_obj"].items():
        if isinstance(ot, tuple):
            locked_objects = {x for x in input_objects if ot in go["object_state"][x]['active_locks']}
            # locks = {ot[0]: [], ot[1]: []}
            # for lo in locked_objects:
            #     locks[go["object_state"][lo]["state"]["ocel:type"]].append(lo)
            if locked_objects:
                existing_objs[ot] = len(locked_objects) / 2
        else:
            objs = {x for x in input_objects if go["object_state"][x]["state"]["ocel:type"]}
            if objs:
                existing_objs[ot] = len(objs)

    return existing_objs


def test_input_obj_sufficiency(go, required, situation, input_objects):
    for ot, quantity in required.items():
        if quantity == "all":
            locked_objects = {x for x in input_objects if ot in go["object_state"][x]['active_locks']}
            quant = []
            for o in locked_objects:
                placement = 0 if ot[0] == go["object_state"][o]["state"]["ocel:type"] else 1
                quant.append(len(go["object_state"][o]["locked_objects"][ot[1-placement]]))
            if max(quant):
                return False

        elif quantity == "any":
            if ot not in situation:
                return False
        else:
            if ot not in situation or quantity < situation[ot]:
                return False

    return True


def gather_objects(activity_dict, prev_objects, expired_view, go):
    # 1. check if event is locked
    # 2. if locked, pass locked objects along and check for remaining requirements
    # 3. if not locked choose objects at random from the expired view

    locked_types = [(x, q) for x, q in activity_dict["input_obj"].items() if isinstance(x, tuple)]
    free_types = [(x, q) for x, q in activity_dict["input_obj"].items() if not isinstance(x, tuple)]

    exp_view = expired_view[activity_dict["name"]]

    unused_locked = set()
    used_obj = set()

    if activity_dict["object_relation"] == "locked":
        for ot, quantity in locked_types:
            locked_objects = {x for x in prev_objects if ot in go["object_state"][x]['active_locks']}
            locks = {ot[0]: [], ot[1]: []}
            for lo in locked_objects:
                locks[go["object_state"][lo]["state"]["ocel:type"]].append(lo)
            locked_objects = []
            for l1 in locks[ot[0]]:
                for l2 in locks[ot[1]]:
                    locked_objects.append((l1, l2))

            if locked_objects:
                if quantity == "all":
                    used_obj.update(*locked_objects)
                elif isinstance(quantity, int):
                    to_use = min(random.randint(1, quantity), len(locked_objects))
                    used_obj.update(*locked_objects[:to_use])
                    unused_locked.update(*locked_objects[to_use:])
                else:
                    to_use = random.randint(1, len(locked_objects)-1)
                    used_obj.update(*locked_objects[:to_use])
                    unused_locked.update(*locked_objects[to_use:])

    for ot, quantity in free_types:
        ot_list = [x for x in exp_view if x in go["ot_dir"][ot]]
        random.shuffle(ot_list)
        if ot_list:
            used_obj |= set(ot_list[:quantity])

    return used_obj, unused_locked


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
        global_objects |= global_event[1][2]

    for expired_event in expired_events:
        process, activity_name, remaining_objects, _ = expired_event[1]
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
    if isinstance(prob_tuple, tuple):
        f, params = prob_tuple
        return f.rvs(**params)
    else:
        return random.choice(list(prob_tuple))


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
    "properties": {"_completed": {False}}
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
    "input_obj": {("order", "item"): "all"},
    "output_obj": None,
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
    "input_obj": {"employee": 1, ("order", "item"): 1},
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
    "input_obj": {"system": 1, ("order", "item"): "all"},
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
    "input_obj": {"employee": 1, ("order", "item"): 1},
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
    "input_obj": {"employee": 1, ("item", "package"): 1},
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
    "input_obj": {("package", "route"): "All"},
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
    "input_obj": {("package", "route"): "Any"},
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
    "input_obj": {("package", "route"): "Any"},
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
    "input_obj": {("package", "route"): "All"},
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
    "input_obj": {("package", "route"): "all"},
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
    global_options = {"timeframe": (datetime(2022, 5, 1, 8),
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
