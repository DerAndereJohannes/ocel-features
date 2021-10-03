import ocel
import ocel_features.variants.object_based as ob
from ocel_features.util.conv_dict_pandas import conv_dict_to_pandas
from ocel_features.util.ocel_alterations import omap_list_to_set


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    omap_list_to_set(log)

    f, v = ob.extract_object_features(log,
                                      ['o1', 'o2', 'o3'], ['neighbour_count'])

    df, row_oids = conv_dict_to_pandas(f, v)

    print(row_oids)
    print(df)


if __name__ == '__main__':
    main()
