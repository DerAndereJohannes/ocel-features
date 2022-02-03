import ocel
import ocel_features.util.local_helper as lh
import ocel_features.analysis.plot_graphs as pg
from pprint import pprint
from ocel_features.util.ocel_helper import omap_list_to_set
from ocel_features.util.multigraph import Relations, \
     create_object_centric_graph


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    omap_list_to_set(log)
    rels = [Relations.MINION]
    rel_graph = create_object_centric_graph(log, rels)

    print(list(rel_graph.nodes))
    print(list(rel_graph.edges))
    localities = lh.obj_relationship_localities(rel_graph, rels)
    # pprint(localities)
    pprint(lh.unique_relations_to_objects(localities, rels))

    # print([f"{k}, {v}" for k, v in rel_graph.nodes.items()])

    pg.show_graph_plt(rel_graph)


if __name__ == '__main__':
    main()
