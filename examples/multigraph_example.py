import ocel
import networkx as nx
import matplotlib.pyplot as plt
from ocel_features.util.ocel_alterations import omap_list_to_set
import ocel_features.util.multigraph as mg


label_convert = {
    'interacts': 'in',
    'descendant': 'de',
    'ancestor': 'an',
    'cobirth': 'cb',
    'codeath': 'cd',
    'merge': 'me',
    'inheritance': 'ih'
}


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    omap_list_to_set(log)
    rels = [mg.Relations.MERGE, mg.Relations.INTERACTS]
    rel_graph = mg.create_object_centric_graph(log, rels)

    print(list(rel_graph.nodes))
    print(list(rel_graph.edges))

    # print([f"{k}, {v}" for k, v in rel_graph.nodes.items()])

    show_graph_plt(rel_graph)


def show_graph_plt(net):
    weights = [len(net.get_edge_data(u, v, []))
               + len(net.get_edge_data(v, u, [])) for u, v in net.edges()]
    edge_labels = {}
    for u, v in net.edges():
        if (u, v) not in edge_labels and (v, u) not in edge_labels:
            uv = [label_convert[x] for x in net.get_edge_data(u, v, [])]
            vu = [label_convert[x] for x in net.get_edge_data(v, u, [])]
            edge_labels[u, v] = f'{u}->{v}:{uv}, {vu}'

    plt.figure()
    pos = nx.nx_pydot.graphviz_layout(net)
    nx.draw_networkx_edge_labels(net, pos, edge_labels)
    # nx.draw(net, pos=pos, with_labels=True)
    nx.draw(net, pos=pos, width=weights, with_labels=True)
    plt.show()


if __name__ == '__main__':
    main()
