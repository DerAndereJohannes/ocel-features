import ocel
import ocel_features.util.object_descendants as od
from ocel_features.util.ocel_alterations import omap_list_to_set


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    omap_list_to_set(log)

    net = od.create_obj_descendant_graph(log)

    [print(f'Obj: {k}, {v}') for k, v in od.get_obj_descendants(net).items()]


if __name__ == '__main__':
    main()
