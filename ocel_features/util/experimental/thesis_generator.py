import random, sys, heapq, ocel
from datetime import timedelta, datetime
from scipy.stats import norm, uniform, weibull_min, bernoulli, arcsine, expon
from copy import copy
from pprint import pprint


def generate_sample_log(config):
    log = generate_empty_log()
    log["ocel:events"] = []

    execute_order_stage(config, log)
    # pprint(log)
    print(len(log["ocel:objects"]), len(log["ocel:events"]))

    execute_package_stage(config, log)
    print(len(log["ocel:objects"]), len(log["ocel:events"]))

    execute_route_stage(config, log)

    # sys_obj = generate_empty_object(system)
    log['ocel:objects']['SYS'] = generate_empty_object("system")
    sys_event = generate_default_event(create_system)
    sys_event["ocel:timestamp"] = datetime.min
    sys_event["ocel:omap"] = {'SYS'}
    log["ocel:events"].append(sys_event)

    event_dict = {}
    log["ocel:events"] = sorted(log["ocel:events"], key=lambda x: x["ocel:timestamp"])
    for i, e in enumerate(log["ocel:events"]):
        e["ocel:omap"] = list(e["ocel:omap"])
        event_dict[f'e{i}'] = e

    for o, v in log["ocel:objects"].items():
        if "ocel:omap" in v:
            del v["ocel:omap"]

    log['ocel:events'] = event_dict
    print(len(log["ocel:objects"]), len(log["ocel:events"]), "\n")

    print(f'order #: {len([x for x in log["ocel:objects"] if x[0] == "o"])}')
    print(f'item #: {len([x for x in log["ocel:objects"] if x[0] == "i"])}')
    print(f'package #: {len([x for x in log["ocel:objects"] if x[0] == "p"])}')
    print(f'route #: {len([x for x in log["ocel:objects"] if x[0] == "r"])}\n')

    items_packed = set()
    for k, v in log["ocel:events"].items():
        if v["ocel:activity"] == "pack items":
            items_packed |= {x for x in v["ocel:omap"] if x[0] == 'i'}
    print("items packed:", len(items_packed))

    packages_delivered = set()
    items_delivered = set()
    failed_deliveries = set()
    delivered_events = set()
    for k, v in log["ocel:events"].items():
        if v["ocel:activity"] == "load package":
            packages_delivered |= {x for x in v["ocel:omap"] if x[0] == 'p'}
            items_delivered |= {x for x in v["ocel:omap"] if x[0] == 'i'}
        elif v["ocel:activity"] == "fail delivery":
            failed_deliveries.add(k)
        elif v["ocel:activity"] == "deliver package":
            delivered_events.add(k)

    print("packages delivered:", len(packages_delivered))
    print("items delivered:", len(items_delivered), "\n")
    print("fail delivery event #:", len(failed_deliveries))
    print("success delivery event #:", len(delivered_events))

    if len(sys.argv) == 2:
        ocel.export_log(log, sys.argv[1])

    # import matplotlib.pyplot as plt
    # plt.hist([x["ocel:timestamp"] for x in log["ocel:events"]])
    # plt.show()


def execute_order_stage(config, log):
    events = []
    objects = {}
    order_count = 0
    item_count = 0
    log_start, log_end = config["start"], config["end"]
    current_day = log_start.weekday()
    employees = ["Jenny", "Jacob", "Julia", "Josh"]
    order_employees = {}
    for e in employees:
        new_empl = generate_empty_object("employee")
        new_empl["ocel:ovmap"] = generate_object_ovmap(employee)
        # add unique event
        new_empl_event = generate_default_event(create_employee)
        new_empl_event["ocel:timestamp"] = datetime.min
        new_empl_event["ocel:omap"] = {e}
        events.append(new_empl_event)
        # add unique object
        order_employees[e] = new_empl
        objects[e] = new_empl

    day_start, day_end = config["day_order_timeframe"]
    timerange = (day_end - day_start).total_seconds()

    for day in range((log_end - log_start).days):

        if current_day < 7:
            current_day += 1
            if current_day == 7:
                continue
        else:
            current_day = 1
            continue

        quantity = int(compute_rvs(config["day_order_quantity"][0]))
        order_start_times = [arcsine.rvs(scale=timerange)
                             for _ in range(quantity)]

        for times in order_start_times:
            # generate order objects
            pick_employee = random.choices(employees, weights=(7, 9, 11, 3))[0]
            order_objects = {"order": None, "item": set(), "employee": pick_employee}
            new_order = generate_empty_object("order")
            new_order["ocel:ovmap"] = generate_object_ovmap(order)
            new_order_id = f'o{order_count}'
            objects[new_order_id] = new_order
            order_objects["order"] = new_order_id
            order_count += 1

            item_number = 0
            if new_order["ocel:ovmap"]["Priority"] == 3:
                item_number = random.choices([1, 2, 3], weights=(10, 5, 1))[0]
            elif new_order["ocel:ovmap"]["Priority"] == 2:
                item_number = random.choices([1, 2, 3], weights=(5, 10, 1))[0]
            else:
                item_number = random.choices([1, 2, 3], weights=(1, 5, 10))[0]

            for i in range(item_number):
                new_item = generate_empty_object("item")
                new_item["ocel:ovmap"] = generate_object_ovmap(item)
                new_item["ocel:ovmap"]['cost($)'] = compute_rvs((norm, {"loc": 500+(500*(0.333*new_order['ocel:ovmap']['Priority'])), "scale": 100}))
                new_item_id = f'i{item_count}'
                objects[new_item_id] = new_item
                order_objects["item"].add(new_item_id)
                item_count += 1

            # generate timestamp for start of order
            timestamp_create = log_start + day_start + timedelta(days=day, seconds=times)

            # create order
            co = generate_default_event(create_order)
            co["ocel:timestamp"] = timestamp_create
            co["ocel:vmap"] = generate_object_ovmap(create_order)
            co["ocel:omap"] = {"SYS", order_objects["order"]} | order_objects["item"]
            events.append(co)

            # accept order
            timestamp_accept = time_in_bounds(timestamp_create + timedelta(hours=max(norm.rvs(loc=0.75, scale=0.15), 0.1)), day_start, day_end)
            ao = generate_default_event(accept_order)
            ao["ocel:timestamp"] = timestamp_accept
            ao["ocel:vmap"] = generate_object_ovmap(accept_order)
            ao["ocel:omap"] = {order_objects["employee"], order_objects["order"]} | order_objects["item"]
            events.append(ao)

            # send invoice
            timestamp_send = time_in_bounds(timestamp_accept + timedelta(minutes=max(norm.rvs(loc=7, scale=2), 0.1)), day_start, day_end)
            so = generate_default_event(send_invoice)
            so["ocel:timestamp"] = timestamp_send
            so["ocel:vmap"] = generate_object_ovmap(send_invoice)
            so["ocel:omap"] = {order_objects["employee"], order_objects["order"]}
            events.append(so)

            # receive payment
            timestamp_pay = timestamp_send + timedelta(days=max(norm.rvs(loc=0.75, scale=0.15), 0.1))
            rp = generate_default_event(receive_payment)
            rp["ocel:timestamp"] = timestamp_pay
            rp["ocel:vmap"] = generate_object_ovmap(receive_payment)
            rp["ocel:omap"] = {"SYS", order_objects["order"]}
            events.append(rp)

            # check avalability and pick item
            timestamp_check = time_in_bounds(timestamp_accept + timedelta(hours=max(norm.rvs(loc=2, scale=0.5), 0.1)), day_start, day_end)
            item_order = list(order_objects["item"])
            random.shuffle(item_order)
            for item_iter in item_order:
                timestamp_check = time_in_bounds(timestamp_check + timedelta(minutes=max(norm.rvs(loc=5, scale=2), 0.1)), day_start, day_end)
                ca = generate_default_event(check_availability)
                ca["ocel:timestamp"] = timestamp_check
                ca["ocel:vmap"] = generate_object_ovmap(check_availability)
                ca["ocel:omap"] = {order_objects["employee"], order_objects["order"], item_iter}
                events.append(ca)

                timestamp_check = time_in_bounds(timestamp_check + timedelta(minutes=max(norm.rvs(loc=5, scale=2), 0.1)), day_start, day_end)
                pi = generate_default_event(pick_item)
                pi["ocel:timestamp"] = timestamp_check
                pi["ocel:vmap"] = generate_object_ovmap(pick_item)
                pi["ocel:omap"] = {order_objects["employee"], order_objects["order"], item_iter}
                events.append(pi)

    log["ocel:events"].extend(events)
    log["ocel:objects"] |= objects


def execute_package_stage(config, log):
    events = []
    objects = {}
    package_count = 0
    log_start, log_end = config["start"], config["end"]
    current_day = log_start.weekday()
    employees = ["Pazu", "Penelope"]
    order_employees = {}
    for e in employees:
        new_empl = generate_empty_object("employee")
        new_empl["ocel:ovmap"] = generate_object_ovmap(employee)
        # add unique event
        new_empl_event = generate_default_event(create_employee)
        new_empl_event["ocel:timestamp"] = datetime.min
        new_empl_event["ocel:omap"] = {e}
        events.append(new_empl_event)
        # add unique object
        order_employees[e] = new_empl
        objects[e] = new_empl

    day_start, day_end = config["day_package_timeframe"]
    timerange = (day_end - day_start).total_seconds()

    # gather neccessary item info
    items_info = [(x["ocel:timestamp"], x["ocel:omap"])
                  for x in log["ocel:events"]
                  if x["ocel:activity"] == check_availability["name"]]

    item_sort = []
    for i in items_info:
        curr_order = [x for x in i[1] if x[0] == 'o'][0]
        for obj in i[1]:
            if obj[0] == 'i':
                item_sort.append((i[0], (curr_order, obj)))

    # start creation
    for day in range((log_end - log_start).days):
        if current_day < 7:
            current_day += 1
            if current_day == 7:
                continue
        else:
            current_day = 1
            continue

        quantity = int(compute_rvs(config["day_package_quantity"][0]))

        package_start_times = [arcsine.rvs(scale=timerange)
                               for _ in range(quantity)]

        for times in package_start_times:
            # generate timestamp for start of order
            timestamp_pack = log_start + day_start + timedelta(days=day, seconds=times)
            pick_employee = random.choices(employees, weights=(9, 7))[0]

            if datetime(2022, 5, 1) < timestamp_pack < datetime(2022, 6, 14) and pick_employee == 'Penelope':
                if not bernoulli.rvs(0.3):
                    continue

            # extract oldest ready items
            item_sort.sort()
            expired_length = 0
            for i, item_iter in enumerate(item_sort):
                if item_iter[0] >= timestamp_pack:
                    expired_length = i
                    break

            if expired_length == 0:
                continue

            expired_items = item_sort[:expired_length]
            item_sort = item_sort[expired_length:]

            package_objects = {"order": None, "item": set(), "employee": pick_employee}
            new_package = generate_empty_object("package")
            new_package["ocel:ovmap"] = generate_object_ovmap(package)
            new_package_id = f'p{package_count}'
            objects[new_package_id] = new_package
            package_objects["package"] = new_package_id
            package_count += 1

            # sort items based on destination, age and priority
            dest = {k: [] for k in range(10)}
            for exp_item in expired_items:
                curr_order = log["ocel:objects"][exp_item[1][0]]["ocel:ovmap"]
                time_prio = 0.3 * min(((((timestamp_pack - exp_item[0]).seconds / 3600) / 168) * 3), 3)
                prio_prio = 0.7 * curr_order["Priority"]
                dest[curr_order["Destination"]].append((time_prio + prio_prio, exp_item[1][1]))

            dest_prio = {k: 0 for k in range(10)}
            for k in dest_prio:
                for inst in dest[k]:
                    dest_prio[k] += inst[0]
            max_prio_dest = max(dest_prio, key=dest_prio.get)
            sorted_prio = sorted(dest[max_prio_dest], key=lambda x: x[0], reverse=True)
            items_add = set()
            items_weight = 0.0
            package_prio = 0.0
            for prio_item in sorted_prio:
                item_id = prio_item[1]
                item_weight = log["ocel:objects"][prio_item[1]]["ocel:ovmap"]["weight(kg)"]

                if items_weight + item_weight < config["package_max_weight"]:
                    package_prio += prio_item[0]
                    items_add.add(item_id)
                    items_weight += item_weight

            package_objects["item"] = items_add

            # readd unused items_add
            for exp_item in expired_items:
                if exp_item[1][1] not in items_add:
                    item_sort.append(exp_item)

            objects[package_objects["package"]]["ocel:ovmap"]["Priority"] = package_prio
            objects[package_objects["package"]]["ocel:ovmap"]["Capacity"] = len(items_add)
            objects[package_objects["package"]]["ocel:ovmap"]["Destination"] = max_prio_dest
            objects[package_objects["package"]]["ocel:ovmap"]["weight(kg)"] = items_weight

            # pack items
            pi = generate_default_event(pack_items)
            pi["ocel:timestamp"] = timestamp_pack
            pi["ocel:vmap"] = generate_object_ovmap(pack_items)
            pi["ocel:omap"] = {package_objects["package"], package_objects["employee"]} | package_objects["item"]
            events.append(pi)

            # store package
            timestamp_store = time_in_bounds(timestamp_pack + timedelta(hours=max(norm.rvs(loc=1, scale=0.2), 0.1)), day_start, day_end)
            sp = generate_default_event(store_package)
            sp["ocel:timestamp"] = timestamp_store
            sp["ocel:vmap"] = generate_object_ovmap(store_package)
            sp["ocel:omap"] = {package_objects["package"], package_objects["employee"]} | package_objects["item"]
            events.append(sp)

    log["ocel:events"].extend(events)
    log["ocel:objects"] |= objects


def execute_route_stage(config, log):
    events = []
    objects = {}
    route_count = 0
    log_start, log_end = config["start"], config["end"]
    current_day = log_start.weekday()
    employees = ["Kiki", "Pat Clifton", "Pete", "Jack Danger", "Resetti", "Norton"]
    employee_perf = {"Kiki": 10, "Pat Clifton": 9, "Pete": 7, "Jack Danger": 4, "Resetti": 11, "Norton": 5}
    order_employees = {}
    for e in employees:
        new_empl = generate_empty_object("employee")
        new_empl["ocel:ovmap"] = generate_object_ovmap(employee)
        # add unique event
        new_empl_event = generate_default_event(create_employee)
        new_empl_event["ocel:timestamp"] = datetime.min
        new_empl_event["ocel:omap"] = {e}
        events.append(new_empl_event)
        # add unique object
        order_employees[e] = new_empl
        objects[e] = new_empl

    # gather neccessary item info
    seen_packages = set()
    package_info = []
    for ev in log["ocel:events"][::-1]:
        if ev["ocel:activity"] == store_package["name"] and frozenset(ev["ocel:omap"]) not in seen_packages:
            package_items = {i for i in ev["ocel:omap"] if i[0] == 'i'}
            for obj in ev["ocel:omap"]:
                if obj[0] == 'p':
                    package_info.append((ev["ocel:timestamp"], (obj, package_items)))
            seen_packages.add(frozenset(ev["ocel:omap"]))

    day_start, day_end = config["day_route_timeframe"]
    timerange = (day_end - day_start).total_seconds()

    # start creation
    for day in range((log_end - log_start).days):
        if current_day < 7:
            current_day += 1
            if current_day == 7:
                continue
        else:
            current_day = 1
            continue

        quantity = int(compute_rvs(config["day_route_quantity"][0]))

        route_start_times = [arcsine.rvs(scale=timerange)
                             for _ in range(quantity)]

        route_start_times.sort()

        for times in route_start_times:

            timestamp_route = log_start + day_start + timedelta(days=day, seconds=times)
            # extract oldest ready items
            package_info.sort()
            expired_length = 0
            for i, item_iter in enumerate(package_info):
                if item_iter[0] >= timestamp_route:
                    expired_length = i
                    break

            if expired_length == 0:
                continue

            deliver_order = copy(employees)
            pick_employee = random.choices(deliver_order, weights=[employee_perf[x] for x in deliver_order])[0]
            deliver_order.remove(pick_employee)
            route_objects = {"route": None, "package": set(), "employee": pick_employee}
            new_route = generate_empty_object("route")
            new_route["ocel:omap"] = {pick_employee}
            new_route["ocel:ovmap"] = generate_object_ovmap(route)
            new_route_id = f'r{route_count}'
            objects[new_route_id] = new_route
            route_objects["route"] = new_route_id
            route_count += 1

            # generate timestamp for start of order
            expired_items = package_info[:expired_length]
            package_info = package_info[expired_length:]

            # sort items based on destination, age and priority
            dest = {k: [] for k in range(10)}
            for exp_item in expired_items:
                curr_package = log["ocel:objects"][exp_item[1][0]]["ocel:ovmap"]
                time_prio = min(0.4*(((((timestamp_route - exp_item[0]).seconds / 3600) / 168)) * 3), 3)
                prio_prio = 0.6 * curr_package["Priority"]
                dest[curr_package["Destination"]].append((time_prio + prio_prio, exp_item[1]))

            dest_prio = {k: 0 for k in range(10)}
            for k in dest_prio:
                for inst in dest[k]:
                    dest_prio[k] += inst[0]
            max_prio_dest = max(dest_prio, key=dest_prio.get)
            sorted_prio = sorted(dest[max_prio_dest], key=lambda x: x[0], reverse=True)

            packages_add = set()
            package_items = {}
            packages_weight = 0.0
            package_prio = 0.0
            for prio_item in sorted_prio:
                item_id = prio_item[1][0]
                package_weight = log["ocel:objects"][item_id]["ocel:ovmap"]["weight(kg)"]

                if packages_weight + package_weight < config["route_max_weight"]:
                    package_prio += prio_item[0]
                    packages_add.add(item_id)
                    package_items[item_id] = prio_item[1][1]
                    packages_weight += package_weight

            if packages_weight < 50 or package_prio < 30:
                packages_add = {}
                del objects[new_route_id]
            else:
                route_objects["package"] = packages_add
                objects[new_route_id]["ocel:omap"] |= packages_add

            # readd unused items
            for exp_item in expired_items:
                if exp_item[1][0] not in packages_add:
                    package_info.append(exp_item)

            if new_route_id not in objects:
                continue

            # start route
            sr = generate_default_event(start_route)
            sr["ocel:timestamp"] = timestamp_route
            sr["ocel:vmap"] = generate_object_ovmap(start_route)
            sr["ocel:omap"] = {*route_objects["package"], route_objects["employee"], new_route_id}
            events.append(sr)

            # load package
            timestamp_load = time_in_bounds(timestamp_route + timedelta(minutes=max(weibull_min.rvs(c=2,loc=15, scale=5), 2.6)), day_start, day_end)
            for i, p in enumerate(route_objects["package"]):
                if i > 0:
                    timestamp_load = time_in_bounds(timestamp_load + timedelta(minutes=max(expon.rvs(loc=2, scale=0.4), 1)), day_start, day_end)
                lp = generate_default_event(load_package)
                lp["ocel:timestamp"] = timestamp_load
                lp["ocel:vmap"] = generate_object_ovmap(load_package)
                lp["ocel:vmap"]['effort'] = max(compute_rvs((norm, {'loc': log['ocel:objects'][p]['ocel:ovmap']['weight(kg)'], 'scale': 0.5})), 1)
                lp["ocel:omap"] = {new_route_id, p, route_objects["employee"]} | package_items[p]
                events.append(lp)

            # to deliver or not to deliver..
            package_failure = []
            timestamp_travel = time_in_bounds(timestamp_load + timedelta(hours=weibull_min.rvs(c=4,loc=1, scale=0.5)), day_start, day_end)
            for i, p in enumerate(route_objects["package"]):
                if i > 0:
                    timestamp_travel = time_in_bounds(timestamp_travel + timedelta(minutes=weibull_min.rvs(c=1,loc=10, scale=2)), day_start, day_end)

                # probability of failure
                prio_addition = (log["ocel:objects"][p]["ocel:ovmap"]["Priority"] / log["ocel:objects"][p]["ocel:ovmap"]["Capacity"]) / 3 * 0.3
                prio_dest = 0.5
                package_destination = int(log['ocel:objects'][p]['ocel:ovmap']['Destination'])

                if package_destination <= 3:
                    prio_dest *= 1
                elif 3 < package_destination <= 5:
                    prio_dest *= 0.8
                else:
                    prio_dest *= 0.5

                prio_bern = 0.2 + prio_dest + prio_addition
                if bernoulli.rvs(prio_bern):
                    # success
                    dp = generate_default_event(deliver_package)
                    dp["ocel:timestamp"] = timestamp_travel
                    dp["ocel:vmap"] = generate_object_ovmap(deliver_package)
                    dp["ocel:omap"] = {p, route_objects["employee"], new_route_id}
                    events.append(dp)
                else:
                    # fail
                    fd = generate_default_event(fail_delivery)
                    fd["ocel:timestamp"] = timestamp_travel
                    fd["ocel:vmap"] = generate_object_ovmap(fail_delivery)
                    fd["ocel:omap"] = {p, route_objects["employee"], new_route_id}
                    events.append(fd)
                    package_failure.append(p)


            timestamp_return = time_in_bounds(timestamp_travel + timedelta(hours=weibull_min.rvs(c=4, loc=1, scale=0.5)), day_start, day_end)
            for i, p in enumerate(package_failure):
                if i > 0:
                    timestamp_return = time_in_bounds(timestamp_travel + timedelta(minutes=weibull_min.rvs(c=1, loc=2, scale=0.1)), day_start, day_end)
                sp = generate_default_event(store_package)
                sp["ocel:timestamp"] = timestamp_return
                sp["ocel:vmap"] = generate_object_ovmap(store_package)
                sp["ocel:omap"] = {p, route_objects["employee"], new_route_id}
                events.append(sp)

            timestamp_end = time_in_bounds(timestamp_return + timedelta(minutes=weibull_min.rvs(c=1, loc=1, scale=0.8)), day_start, day_end)
            er = generate_default_event(end_route)
            er["ocel:timestamp"] = timestamp_end
            er["ocel:vmap"] = generate_object_ovmap(end_route)
            er["ocel:omap"] = {route_objects["employee"], new_route_id}
            events.append(er)

            # readd unused items
            for exp_item in expired_items:
                if exp_item[1][0] in package_failure:
                    package_info.append(exp_item)

    log["ocel:events"].extend(events)
    log["ocel:objects"] |= objects


def time_in_bounds(time, lower, upper):
    lower_hour = (datetime.min + lower).hour
    upper_hour = (datetime.min + upper).hour
    if lower_hour <= time.hour < upper_hour:
        return time
    elif time.hour < lower_hour:
        return time + timedelta(hours=lower_hour - time.hour + 0.2)
    else:
        return time + timedelta(hours=24 - time.hour + lower_hour + 0.2)


def generate_object_ovmap(otype):
    ovmap = {}
    if otype["properties"]:
        for k, v in otype["properties"].items():
            ovmap[k] = compute_rvs(v)
    return ovmap


def generate_default_event(act):
    return {"ocel:activity": act["name"], "ocel:timestamp": datetime.min,
            "ocel:omap": set(), "ocel:vmap": {}}


def generate_empty_object(otype):
    return {"ocel:type": otype, "ocel:ovmap": {}}


def generate_empty_log():
    log = {
           'ocel:global-log': {'ocel:version': '0.1',
                               'ocel:ordering': 'timestamp',
                               'ocel:attribute-names': [""],
                               'ocel:object-types': ["system", "employee",
                                                     "order", "item",
                                                     "package", "route"]
                               },
           'ocel:global-object': {},
           'ocel:global-event': {},
           'ocel:events': {},
           'ocel:objects': {}
           }

    return log


def compute_rvs(prob_tuple):
    if isinstance(prob_tuple, tuple):
        f, params = prob_tuple
        return f.rvs(**params)
    else:
        return random.choice(list(prob_tuple))


generation_config = {
            "start": datetime(2022, 1, 1),
            "end": datetime(2024, 1, 1),
            "day_order_quantity": ((norm, {"loc": 90, "scale": 10}), (arcsine, {})),
            "day_order_timeframe": (timedelta(hours=8), timedelta(hours=18)),
            "day_package_quantity": ((expon, {"loc": 50}), (arcsine, {})),
            "day_package_timeframe": (timedelta(hours=12), timedelta(hours=18)),
            "package_max_weight": 15,
            "day_route_quantity": ([4, 5, 6], (arcsine, {})),
            "day_route_timeframe": (timedelta(hours=6), timedelta(hours=12)),
            "route_max_weight": 200
        }


# object types
employee = {
    "name": "employee",
    "properties": {
        "rating": (norm, {"loc": 100, "scale": 3}),
    }
}

system = {
    "name": "system",
    "properties": None
}


order = {
    "name": "order",
    "properties": {
        "Priority": [1, 2, 3],
        "Destination": [x for x in range(10)],
        "Client": [x for x in range(10)]*4+[x for x in range(30)]
    }
}

item = {
    "name": "item",
    "properties": {
        "weight(kg)": (expon, {"loc": 3, "scale": 0.7}),
        "Category": ['Electronics', 'Raw Material', 'Chemicals']
    }
}

package = {
    "name": "package",
    "properties": {"material": ['wood', 'card', 'plastic']}
}

route = {
    "name": "route",
    "properties": None
}

# activities

setup_order = {
    "name": "setup order",
    "properties": None
}

create_order = {
    "name": "create order",
    "properties": {
        'effort': (norm, {'loc': 5, 'scale': 0.8}),
        'cost': [1]
    }
}

accept_order = {
    "name": "accept order",
    "properties": {
        'effort': (norm, {'loc': 2, 'scale': 0.2}),
        'cost': [1]
    }
}

check_availability = {
    "name": "check availability",
    "properties": {
        'effort': (norm, {'loc': 2, 'scale': 0.2}),
    }
}

send_invoice = {
    "name": "send invoice",
    "properties": {
        'effort': (norm, {'loc': 5, 'scale': 0.5}),
        'cost': [5]
    }
}

receive_payment = {
    "name": "receive payment",
    "properties": {
        'payment type': {'credit card', 'debit card',
                         'bank transfer'},
        'effort': (norm, {'loc': 1, 'scale': 0.1}),
        'cost': [3]
                   }
}

pick_item = {
    "name": "pick item",
    "properties": {
        'effort': (norm, {'loc': 10, 'scale': 3})
        }
}

pack_items = {
    "name": "pack items",
    "properties": {
        'effort': (norm, {'loc': 30, 'scale': 5}),
        'cost': [5]
    }
}

store_package = {
    "name": "store package",
    "properties": {
        'effort': (norm, {'loc': 10, 'scale': 1}),
        'cost': [5]
    }
}

setup_route = {
    "name": "setup route",
    "properties": {
        # 'cost': [150]
        }
}

start_route = {
    "name": "start route",
    "properties": {
        'effort': (norm, {'loc': 2, 'scale': 0.2}),
        'cost': [150]
    }
}

load_package = {
    "name": "load package",
    "properties": None
}

fail_delivery = {
    "name": "fail delivery",
    "properties": {
        'effort': (norm, {'loc': 10, 'scale': 0.5}),
        'cost': [50]
    }
}

deliver_package = {
    "name": "deliver package",
    "properties": {
        'effort': (norm, {'loc': 7, 'scale': 0.8}),
        'cost': [10]
    }
}

unload_package = {
    "name": "unload package",
    "properties": {
        'effort': (norm, {'loc': 10, 'scale': 1}),
        'cost': [10]
    }
}

end_route = {
    "name": "end route",
    "properties": {
        'effort': (norm, {'loc': 1, 'scale': 0.1})
    }
}

create_employee = {
    "name": "create employee",
    "properties": None
}
create_system = {
    "name": "create system",
    "properties": None
}

if __name__ == "__main__":
    generate_sample_log(generation_config)
