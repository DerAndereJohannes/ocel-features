import ocel
from ocel_features.util.ocel_alterations import omap_list_to_set
import ocel_features.util.global_attribute_names as goan


def main():
    log = ocel.import_log('logs/min.jsonocel')
    omap_list_to_set(log)

    print('all objs:', goan.get_common_attribute_names(log))
    print('item obj:', goan.get_common_attribute_names(log, 'item'))


if __name__ == '__main__':
    main()
