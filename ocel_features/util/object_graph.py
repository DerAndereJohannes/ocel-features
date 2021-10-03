import ocel
import networkx as nx


def create_object_graph(log):
    ocel_events = ocel.get_events(log)
    ocel_objects = ocel.get_objects(log)
    ids_checked = set()
    obj_net = nx.Graph()

    for event_id in ocel_events:
        event = ocel_events[event_id]
        for oid in event['ocel:omap']:

            # add all new edges between selected
            for oid2 in event['ocel:omap']:
                if oid is not oid2 and not obj_net.has_edge(oid, oid2):
                    obj_net.add_edge(oid, oid2)

            # add first occured time and event attribute to the node
            if oid not in ids_checked:
                ids_checked.add(oid)
                obj_net.nodes[oid]['first_occurance'] = event['ocel:timestamp']
                obj_net.nodes[oid]['first_event'] = event_id
                obj_net.nodes[oid]['object_events'] = list()
                obj_net.nodes[oid]['type'] = ocel_objects[oid]['ocel:type']

            obj_net.nodes[oid]['object_events'].append(event_id)

    return obj_net
