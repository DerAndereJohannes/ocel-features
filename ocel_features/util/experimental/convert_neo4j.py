from neo4j import GraphDatabase
import networkx as nx
from datetime import timedelta
from copy import copy


class neo4j_exporter:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_nodes(self, graph):
        exe_list = ["CREATE "]

        for n in graph.nodes():
            exe_list.append(f"(:Object:{graph.nodes[n]['type']}"
                            f" {create_node_str(graph.nodes[n])}),")

        exe_list[-1] = exe_list[-1][:-1]  # remove last comma

        import_nodes_query = ''.join(exe_list)
        with self.driver.session() as session:
            session.run(import_nodes_query)

    def import_relations(self, graph):
        exe_list = []

        for e in graph.edges():
            src = e[0]
            tar = e[1]
            rel = graph.edges[e]['rel']
            exe_list.append(f"MATCH (a:Object), (b:Object) "
                            f"WHERE a.id='{src}' AND b.id='{tar}' "
                            f"CREATE (a)-[:{rel}]->(b)")
        # print(exe_list)
        with self.driver.session() as session:
            for q in exe_list:
                session.run(q)


def create_node_str(n_dict):
    return f"{{id: '{n_dict['id']}'," \
        f"first_occurance: {n_dict['first_occurance']}}}"


def convert_log_to_labeled_graph(log):
    # setup
    events = log['ocel:events']
    objects = log['ocel:objects']
    graph = nx.Graph()

    # add all nodes
    graph.add_nodes_from([(k, v) for k, v in events.items()])
    graph.add_nodes_from([(k, v) for k, v in objects.items()])

    # add all edges
    for e, v in events.items():
        for o in v['ocel:omap']:
            graph.add_edge(e, o)

        graph.nodes[e].pop('ocel:omap')

    return graph


def convert_labeled_graph_to_log(graph):
    log = {'ocel:objects': {},
           'ocel:events': {},
           'ocel:global-log': {'ocel:version': '0.1',
                               'ocel:ordering': 'timestamp',
                               'ocel:attribute-names': set(),
                               'ocel:object-types': set()}
           }
    for n in graph.nodes():
        # if the node is an object or an event
        if 'ocel:type' in graph.nodes[n]:  # it is an object
            log['ocel:objects'][n] = graph.nodes[n]
            log['ocel:global-log']['ocel:object-types'].add(
                graph.nodes[n]['ocel:type'])
            log['ocel:global-log']['ocel:attribute-names'].update(
                {attr for attr in graph.nodes[n]['ocel:ovmap']})
        else:  # it is an event
            e_dict = graph.nodes[n]
            e_dict['ocel:omap'] = [t[1] for t in nx.edges(graph, n)]
            # print(n, e_dict)
            log['ocel:global-log']['ocel:attribute-names'].update(
                {attr for attr in e_dict['ocel:vmap']})
            log['ocel:events'][n] = e_dict

    # final cleanup
    log['ocel:global-log']['ocel:attribute-names'] = list(
        log['ocel:global-log']['ocel:attribute-names'])
    log['ocel:global-log']['ocel:object-types'] = list(
        log['ocel:global-log']['ocel:object-types'])
    log['ocel:events'] = sorted(
        log['ocel:events'].items(), key=lambda x: x[1]['ocel:timestamp'])

    return log


def event_directly_follows(log):
    o_dict = {o: [] for o in log['ocel:objects']}
    graph = nx.DiGraph()
    graph.add_nodes_from([(k, v) for k, v in log['ocel:events'].items()])

    # create object events
    for e, v in log['ocel:events'].items():
        for o in v['ocel:omap']:
            # if there is already an event in the object events
            if o_dict[o]:
                graph.add_edge(o_dict[o][-1], e)

            o_dict[o].append(e)

    return graph


def split_by_time(log, start_time, time_delta):
    end_time = start_time + time_delta
    log_events = []

    for e, v in log['ocel:events'].items():
        if start_time <= v['ocel:timestamp'] <= end_time:
            log_events.append(e)
        elif log_events:
            # if it is done iterating
            return log_events

    return log_events


def split_by_days_of_week(log, days):
    log_events = {}

    for e, v in log['ocel:events'].items():
        e_time = v['ocel:timestamp']
        week_start = (e_time - timedelta(days=e_time.weekday())).date()

        # only accept wanted days of week
        if e_time.weekday() in days:
            if week_start not in log_events:
                log_events[week_start] = {'ocel:events': [],
                                          'ocel:objects': set()}
            log_events[week_start]['ocel:events'].append(e)
            log_events[week_start]['ocel:objects'].update(
                log['ocel:events'][e]['ocel:omap'])

    return log_events


def get_weekly_graphs(log, days):
    week_logs = split_by_days_of_week(log, days)
    week_graphs = {}

    for k, v in week_logs.items():
        week_log = copy(log)
        week_log['ocel:events'] = {e: log['ocel:events'][e]
                                   for e in v['ocel:events']}
        week_log['ocel:objects'] = {o: log['ocel:objects'][o]
                                    for o in v['ocel:objects']}
        week_graphs[k] = convert_log_to_labeled_graph(week_log)

    return week_graphs
