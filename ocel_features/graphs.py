import os
from ocel_features.util.multigraph import Relations, all_relations, \
 create_object_centric_graph, import_multigraph, export_multigraph
from ocel_features.util.experimental.convert_to_n4j import \
    convert_log_to_labeled_graph, convert_labeled_graph_to_log, \
    event_directly_follows

DEFAULT_RELATIONS = {Relations.DESCENDANTS}
DEFAULT_PATH = os.getcwd() + os.sep


# Object graphs
def object_centric_graph(log, relations=DEFAULT_RELATIONS):
    """Creates an Object centric directed graph based on the object-object
    relations that are relevant.

    Args:
        log (Dict): Object-centric event log.
        relations (Set[Relations], optional): Set of interested relations.
        Defaults to DEFAULT_RELATIONS.

    Returns:
        DiGraph: Networkx DiGraph consisting of the object-object relation
        graph.
    """
    return create_object_centric_graph(log, relations)


def list_relations():
    return all_relations()


def read_object_centric_graph(path):
    """Retrieves an object-centric directed graph from the given path.

    Args:
        path (String/Path): Path to the file

    Returns:
        DiGraph: Networkx Digraph of the object-object relation graph.
    """
    return import_multigraph(path)


def write_object_centric_graph(graph, path=f'{DEFAULT_PATH}ocg.gml'):
    """Writes an object-centric directed graph to disk for later use. This is
    very useful if a graph takes a long time to generate.

    Args:
        graph (DiGraph): Object-object networkx Graph
        path (String, optional): Path to save the file.
        Defaults to f'{DEFAULT_PATH}ocg.gml'.
    """
    export_multigraph(graph, path)


# Event graphs
def event_df(log):
    """Create a directly follows graph based on the events on the graph.
    Rather than being based on the events ordered by id, the edges are based
    on the directly follows of each individual object events series.

    Args:
        log (Dict): Object-centric event log

    Returns:
        DiGraph: OCEL event directly follows graph
    """
    return event_directly_follows(log)


# Object-event graphs
def log_to_graph(log):
    """Generates a basic object-event graph based on which objects participate
    in which events. Ie. if object o1 participates in e1, there is an
    undirected edge between o1 and e1.

    Args:
        log (Dict): Object-centric event log

    Returns:
        Graph: Networkx graph based on the input OCEL.
    """
    return convert_log_to_labeled_graph(log)


def graph_to_log(graph):
    """Regenerates the ocel log object based on the connections between events
    and objects. Other metadata is also extracted.

    Args:
        graph (Graph): Networkx graph containing object-event information.

    Returns:
        Dict: Object-centric event log.
    """
    return convert_labeled_graph_to_log(graph)
