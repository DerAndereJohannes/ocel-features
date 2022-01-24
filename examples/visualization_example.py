import os
import ocel
import matplotlib.pyplot as plt
import networkx as nx
import ocel_features.util.multigraph as m
from ocel_features.util.ocel_helper import omap_list_to_set


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    omap_list_to_set(log)

    net = m.create_object_centric_graph(log)

    show_graph_plt(net)
    # save_graph_graphviz(net)


def save_graph_graphviz(net):
    ag = nx.nx_agraph.to_agraph(net)
    ag.layout('dot')
    ag.draw('test_graph.png')
    os.system('open test_graph.png')


def show_graph_plt(net):
    weights = [len(net.get_edge_data(u, v, []))
               + len(net.get_edge_data(v, u, [])) for u, v in net.edges()]
    edge_labels = {}
    for u, v in net.edges():
        if (u, v) not in edge_labels and (v, u) not in edge_labels:
            uv = [m.relation_shorthand(x) for x in net.get_edge_data(u, v, [])]
            vu = [m.relation_shorthand(x) for x in net.get_edge_data(v, u, [])]
            edge_labels[u, v] = f'{u}->{v}:{uv}, {vu}'

    plt.figure()
    pos = nx.nx_pydot.graphviz_layout(net)
    nx.draw_networkx_edge_labels(net, pos, edge_labels)
    # nx.draw(net, pos=pos, with_labels=True)
    nx.draw(net, pos=pos, width=weights, with_labels=True)
    plt.show()


if __name__ == '__main__':
    main()
