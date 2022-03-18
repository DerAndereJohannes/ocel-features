import os
import networkx as nx
import matplotlib.pyplot as plt
import ocel_features.util.multigraph as m


def save_graph_graphviz(net, layout='dot', name='test_graph', labels=False):
    ag = nx.nx_agraph.to_agraph(net)
    ag.layout(layout)
    if labels:
        edge_labels = {e: net.get_edge_data(*e) for e in net.edges()}
        nx.draw_networkx_edge_labels(net, edge_labels)
    ag.draw(f'{name}.png')
    os.system(f'xdg-open {name}.png')


def show_graph_plt(net, labels=False):
    # weights = [len(net.get_edge_data(u, v, []))
    #            + len(net.get_edge_data(v, u, [])) for u, v in net.edges()]
    plt.figure()
    # pos = nx.nx_pydot.graphviz_layout(net)
    pos = nx.spring_layout(net)

    if labels:
        edge_labels = {}
        for u, v in net.edges():
            if (u, v) not in edge_labels and (v, u) not in edge_labels:
                uv = [m.relation_shorthand(x)
                      for x in net.get_edge_data(u, v, [])]
                vu = [m.relation_shorthand(x)
                      for x in net.get_edge_data(v, u, [])]
                edge_labels[u, v] = f'{u}->{v}:{uv}, {vu}'

        nx.draw_networkx_edge_labels(net, pos, edge_labels)
    nx.draw(net, pos=pos, with_labels=True)
    plt.show()


def basic_graphviz(net):
    plt.figure()
    nx.draw(net, with_labels=True)
    plt.show()


def without_isolates(net):
    pass
