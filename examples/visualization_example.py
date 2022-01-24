import ocel
import ocel_features.util.multigraph as m
import ocel_features.analysis.plot_graphs as pg
from ocel_features.util.ocel_helper import omap_list_to_set


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    omap_list_to_set(log)

    net = m.create_object_centric_graph(log)

    pg.show_graph_plt(net)
    # pg.save_graph_graphviz(net)


if __name__ == '__main__':
    main()
