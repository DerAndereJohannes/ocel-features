import ocel
from ocel_features.util.ocel_helper import omap_list_to_set, \
     get_common_attribute_names


def main():
    log = ocel.import_log('logs/min.jsonocel')
    omap_list_to_set(log)

    print('all objs:', get_common_attribute_names(log))
    print('item obj:', get_common_attribute_names(log, 'item'))


if __name__ == '__main__':
    main()
