import ocel
import networkx as nx
from copy import deepcopy


def create_obj_descendant_graph(log):
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
                net.nodes[oid]['first_occurance'] = event['ocel:timestamp']
                net.nodes[oid]['first_event'] = event_id
                net.nodes[oid]['type'] = ocel_objects[oid]['ocel:type']
                net.nodes[oid]['object_events'] = list()
                net.nodes[oid]['descendant'] = set()
                net.nodes[oid]['relative'] = set()

            if len(ids_checked) == len(ocel_objects):
                break

        for oid in event['ocel:omap']:
            net.nodes[oid]['object_events'].append(event_id)
            # add all new edges between selected
            for oid2 in event['ocel:omap']:

                if oid is not oid2 and not net.has_edge(oid, oid2) \
                   and not net.has_edge(oid2, oid):

                    if has_init_node(net, oid, oid2, event_id):
                        if same_init_event(net, oid, oid2) \
                           or same_init_time(net, oid, oid2):
                            net.add_edge(oid, oid2, weight=2)
                            net.add_edge(oid2, oid, weight=2)
                            net.nodes[oid2]['descendant'].add(oid)
                            net.nodes[oid]['descendant'].add(oid2)
                        elif oid == get_younger_obj(net, oid, oid2):
                            net.add_edge(oid, oid2, weight=2)
                            net.nodes[oid]['descendant'].add(oid2)
                            net.nodes[oid2]['relative'].add(oid)
                        else:
                            net.add_edge(oid2, oid, weight=2)
                            net.nodes[oid2]['descendant'].add(oid)
                            net.nodes[oid]['relative'].add(oid2)

    return net


def create_obj_all_graph(log):
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
                net.nodes[oid]['first_occurance'] = \
                    event['ocel:timestamp'].timestamp()
                net.nodes[oid]['first_event'] = event_id
                net.nodes[oid]['type'] = ocel_objects[oid]['ocel:type']
                net.nodes[oid]['id'] = oid
                net.nodes[oid]['descendant'] = set()
                net.nodes[oid]['ancestor'] = set()
                net.nodes[oid]['sibling'] = set()

            if len(ids_checked) == len(ocel_objects):
                break

        for oid in event['ocel:omap']:
            # add all new edges between selected
            for oid2 in event['ocel:omap']:

                if oid is not oid2 and not net.has_edge(oid, oid2) \
                   and not net.has_edge(oid2, oid):
                    if has_init_node(net, oid, oid2, event_id):
                        if same_init_event(net, oid, oid2) \
                           or same_init_time(net, oid, oid2):
                            net.add_edge(oid, oid2, rel="SIBLING_OF")
                            net.add_edge(oid2, oid, rel="SIBLING_OF")
                            net.nodes[oid2]['sibling'].add(oid)
                            net.nodes[oid]['sibling'].add(oid2)
                        elif oid == get_younger_obj(net, oid, oid2):
                            net.add_edge(oid, oid2, rel="ANCESTOR_OF")
                            net.add_edge(oid2, oid, rel="DESCENDANT_OF")
                            net.nodes[oid]['descendant'].add(oid2)
                            net.nodes[oid2]['ancestor'].add(oid)
                        else:
                            net.add_edge(oid, oid2, rel="DESCENDANT_OF")
                            net.add_edge(oid2, oid, rel="ANCESTOR_OF")
                            net.nodes[oid2]['descendant'].add(oid)
                            net.nodes[oid]['ancestor'].add(oid2)

    return net


def same_init_event(net, id_1, id_2):
    return net.nodes[id_1]['first_event'] == net.nodes[id_2]['first_event']


def same_init_time(net, id_1, id_2):
    return net.nodes[id_1]['first_occurance'] \
        == net.nodes[id_2]['first_occurance']


def get_younger_obj(net, id_1, id_2):
    if net.nodes[id_1]['first_occurance'] < net.nodes[id_2]['first_occurance']:
        return id_1
    else:
        return id_2


def has_init_node(net, id_1, id_2, event):
    return net.nodes[id_1]['first_event'] == event \
           or net.nodes[id_2]['first_event'] == event


def is_descendant(net, source, target):
    return target in net.nodes[source]['descendant']


def get_obj_descendants(net, obj_list=None):
    if obj_list is None:
        subnets = {n: {'descendants': {n}, 'relatives': set()}
                   for n in net.nodes}
    else:
        subnets = {n: {'descendants': {n}, 'relatives': set()}
                   for n in obj_list}

    for obj_n in subnets:
        descendants = subnets[obj_n]['descendants']
        old_neigh = descendants
        new_neigh = set()

        while len(new_neigh) != 0 or old_neigh is descendants:
            new_neigh = set()
            for obj in old_neigh:
                for neigh in net.neighbors(obj):
                    if neigh not in descendants:
                        if same_init_event(net, obj, neigh) \
                           or is_descendant(net, obj, neigh):
                            new_neigh.add(neigh)
            descendants = descendants | new_neigh
            old_neigh = new_neigh

        for node in descendants:
            if node is not obj_n:
                in_edges = [x[0] for x in net.in_edges(node)]
                for in_obj in in_edges:
                    if in_obj not in descendants \
                       and is_descendant(net, in_obj, node):
                        subnets[obj_n]['relatives'].add((in_obj, node))

        subnets[obj_n]['descendants'] = descendants

    return subnets


def _GENERAL_generate_obj_lineage_graph(log):
    ocel_events = ocel.get_events(log)
    ocel_objects = ocel.get_objects(log)
    obj_props = {o: {} for o in ocel_objects}
    ids_checked = set()
    net = nx.DiGraph()

    for event_id in ocel_events:
        event = ocel_events[event_id]

        for oid in event['ocel:omap']:
            # add first occured time and event attribute to the node
            if oid not in ids_checked:
                ids_checked.add(oid)
                net.add_node(oid)
                obj_props[oid]['first_event'] = event_id
                obj_props[oid]['first_occurance'] = event['ocel:timestamp']
                obj_props[oid]['type'] = ocel_objects[oid]['ocel:type']
                obj_props[oid]['object_events'] = list()
                obj_props[oid]['dir_descendants'] = set()
                obj_props[oid]['dir_ancestors'] = set()
                obj_props[oid]['siblings'] = set()

            if len(ids_checked) == len(ocel_objects):
                break

        for oid in event['ocel:omap']:
            obj_props[oid]['object_events'].append(event_id)
            # add all new edges between selected
            for oid2 in event['ocel:omap']:

                if oid is not oid2 and not net.has_edge(oid, oid2) \
                   and not net.has_edge(oid2, oid):

                    net.add_edge(oid, oid2)
                    net.add_edge(oid2, oid)
                    if has_init_node(net, oid, oid2, event_id):
                        if same_init_event(net, oid, oid2) \
                           or same_init_time(net, oid, oid2):
                            obj_props[oid2]['siblings'].add(oid)
                            obj_props[oid]['siblings'].add(oid2)
                        elif oid == get_younger_obj(net, oid, oid2):
                            obj_props[oid]['dir_descendants'].add(oid2)
                            obj_props[oid2]['dir_ancestors'].add(oid)
                        else:
                            obj_props[oid2]['dir_descendants'].add(oid)
                            obj_props[oid]['dir_ancestors'].add(oid2)

    return net, obj_props


def get_obj_descendants2(net, obj_list=None):
    if obj_list is None:
        subnets = {n: {'descendants': {n}, 'relatives': set()}
                   for n in net.nodes}
    else:
        subnets = {n: {'descendants': {n}, 'relatives': set()}
                   for n in obj_list}

    for obj_n in subnets:
        subnets[obj_n]['siblings'] = {n for n in net.neighbors(obj_n)
                                      if same_init_event(net, obj_n, n)}
        descendants = subnets[obj_n]['descendants']
        old_neigh = descendants
        new_neigh = set()

        while len(new_neigh) != 0 or old_neigh is descendants:
            new_neigh = set()
            for obj in old_neigh:
                for neigh in net.neighbors(obj):
                    if neigh not in descendants:
                        if is_descendant(net, obj, neigh):
                            new_neigh.add(neigh)
            descendants = descendants | new_neigh
            old_neigh = new_neigh

        for node in descendants:
            in_edges = [x[0] for x in net.in_edges(node)]
            for in_obj in in_edges:
                if in_obj not in descendants \
                   and is_descendant(net, in_obj, node):
                    subnets[obj_n]['relatives'].add((in_obj, node))

        subnets[obj_n]['descendants'] = descendants

    return subnets


def get_obj_ancestors(net, obj_list=None):
    if obj_list is None:
        subnets = {n: {'ancestors': {n}}
                   for n in net.nodes}
    else:
        subnets = {n: {'ancestors': {n}}
                   for n in obj_list}
    sorted_oids = [x for x in net.nodes]
    sorted_oids.sort(key=lambda x: net.nodes[x]['first_occurance'])

    for obj_n in sorted_oids:
        ancestors = subnets[obj_n]['ancestors']
        old_neigh = ancestors
        new_neigh = set()

        while len(new_neigh) != 0 or old_neigh is ancestors:
            new_neigh = set()
            for obj in old_neigh:
                for tup in net.in_edges(obj):
                    neigh = tup[0]
                    if neigh not in ancestors:
                        if len(subnets[neigh]['ancestors']) != 1:
                            new_neigh |= subnets[neigh]['ancestors']
                        elif is_descendant(net, neigh, obj) \
                                and not same_init_event(net, neigh, obj):
                            new_neigh.add(neigh)
            ancestors = ancestors | new_neigh
            old_neigh = new_neigh

        subnets[obj_n]['ancestors'] = ancestors

    return subnets


def enhance_relations(net):
    descendant_trees = get_obj_descendants(net)
    ancestor_trees = get_obj_ancestors(net)

    for obj_n in net.nodes:
        net.nodes[obj_n]['descendant_tree'] = \
            descendant_trees[obj_n]['descendants']
        net.nodes[obj_n]['ancestor_tree'] = ancestor_trees[obj_n]['ancestors']
        net.nodes[obj_n]['descendant_tree'].remove(obj_n)
        net.nodes[obj_n]['ancestor_tree'].remove(obj_n)
        net.nodes[obj_n]['extended'] = descendant_trees[obj_n]['relatives']


def get_lineages(graph):
    # can be combined
    descendants = get_obj_descendants(graph)
    ancestors = get_obj_ancestors(graph)
    lineages = {oid: None for oid in graph.nodes}

    for oid in lineages:
        lineages[oid] = {}
        lineages[oid]['localities'] = descendants[oid]['descendants'] \
            | ancestors[oid]['ancestors']

    return lineages


def _get_localities_recursive(descs):
    localities = {oid: None for oid in descs}
    desc_dict = deepcopy(descs)
    total = len(localities) - 1
    for i, oid in enumerate(localities):
        print(f'{i} out of {total}')
        localities[oid] = dict()
        localities[oid]['localities'], _ = climb_up(
            desc_dict[oid]['relatives'], descs, set())

    return localities


def climb_up(relatives, desc_dict, seen):
    to_add = set()
    if not relatives:
        return set()
    for rel in relatives:
        uarc = rel[0]
        print(uarc)
        if uarc not in seen:

            to_add |= desc_dict[uarc]['descendants']
            seen.add(uarc)
            new_d, seen = climb_up(desc_dict[uarc]['relatives'],
                                   desc_dict, seen)
            to_add |= new_d

    return to_add, seen


def get_unique_localities(localities):
    unique_localitites = []

    for lcl in localities.values():
        if lcl['localities'] not in unique_localitites:
            unique_localitites.append(lcl['localities'])

    return unique_localitites


def get_direct_descendants(net, obj_id):
    return net.neighbors(obj_id)
