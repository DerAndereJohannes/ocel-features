import ocel
import networkx as nx
import matplotlib.pyplot as plt
import ocel_features.util.multigraph as mg
import ocel_features.util.local_helper as lh

from pprint import pprint
from ocel_features.util.ocel_alterations import omap_list_to_set


label_convert = {
    'INTERACTS': 'in',
    'DESCENDANTS': 'de',
    'ANCESTORS': 'an',
    'COBIRTH': 'cb',
    'CODEATH': 'cd',
    'COLIFE': 'cl',
    'MERGE': 'me',
    'INHERITANCE': 'ih',
    'MINION': 'mi',
    'PEELER': 'pe',
    'CONSUMES': 'co'
}


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    omap_list_to_set(log)
    rels = [mg.Relations.ANCESTORS2DESCENDANTS]
    rel_graph = mg.create_object_centric_graph(log, rels)

    print(list(rel_graph.nodes))
    print(list(rel_graph.edges))
    localities = lh.obj_relationship_localities(rel_graph, rels)
    pprint(localities)
    pprint(lh.unique_relations_to_objects(localities, rels))

    # print([f"{k}, {v}" for k, v in rel_graph.nodes.items()])

    show_graph_plt(rel_graph)


def show_graph_plt(net):
    weights = [len(net.get_edge_data(u, v, []))
               + len(net.get_edge_data(v, u, [])) for u, v in net.edges()]
    edge_labels = {}
    for u, v in net.edges():
        if (u, v) not in edge_labels and (v, u) not in edge_labels:
            uv = [mg.relation_shorthand(x) for x in net.get_edge_data(u, v, [])]
            vu = [mg.relation_shorthand(x) for x in net.get_edge_data(v, u, [])]
            # uv = [label_convert[x] for x in net.get_edge_data(u, v, [])]
            # vu = [label_convert[x] for x in net.get_edge_data(v, u, [])]
            edge_labels[u, v] = f'{u}->{v}:{uv}, {vu}'

    plt.figure()
    pos = nx.nx_pydot.graphviz_layout(net)
    nx.draw_networkx_edge_labels(net, pos, edge_labels)
    # nx.draw(net, pos=pos, with_labels=True)
    nx.draw(net, pos=pos, width=weights, with_labels=True)
    plt.show()


if __name__ == '__main__':
    main()
