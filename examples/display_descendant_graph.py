import os
import ocel
import matplotlib.pyplot as plt
import networkx as nx
import ocel_features.util.object_descendants as od
from ocel_features.util.ocel_alterations import omap_list_to_set


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    omap_list_to_set(log)

    net = od.create_obj_descendant_graph(log)

    show_graph_plt(net)


def save_graph_graphviz(net):
    ag = nx.nx_agraph.to_agraph(net)
    ag.layout('dot')
    ag.draw('test_graph.png')
    os.system('open test_graph.png')


def show_graph_plt(net):
    edges = net.edges()
    weights = [net[u][v]['weight'] for u, v in edges]
    plt.figure()
    pos = nx.nx_pydot.graphviz_layout(net)
    nx.draw(net, pos=pos, width=weights, with_labels=True)
    plt.show()


if __name__ == '__main__':
    main()
