import ocel
import networkx as nx
import matplotlib.pyplot as plt
from ocel_features.util.ocel_alterations import omap_list_to_set
import ocel_features.util.multigraph as mg


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    omap_list_to_set(log)
    rels = [mg.Relations.HEIRLOOM] #, mg.Relations.COBIRTH]
    rel_graph = mg.create_object_centric_graph(log, rels)

    print(list(rel_graph.nodes))
    print(list(rel_graph.edges))

    # print([f"{k}, {v}" for k, v in rel_graph.nodes.items()])

    show_graph_plt(rel_graph)


def show_graph_plt(net):
    # edges = net.edges()
    # weights = [net[u][v]['weight'] for u, v in edges]
    plt.figure()
    pos = nx.nx_pydot.graphviz_layout(net)
    nx.draw(net, pos=pos, with_labels=True)
    # nx.draw(net, pos=pos, width=weights, with_labels=True)
    plt.show()


if __name__ == '__main__':
    main()
