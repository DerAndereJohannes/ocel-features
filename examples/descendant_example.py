import ocel
from ocel_features.algorithm import apply as extract
from ocel_features.util.ocel_alterations import omap_list_to_set


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    omap_list_to_set(log)

    df, row_oids = extract(log, 'descendant_based',
                           ['o1', 'o2', 'o3'],
                           ['direct_descendant_count', 'descendant_type_count',
                            'total_descendant_count',
                            'descendant_relative_ratio', 'asdf'])

    print('df row IDs:', row_oids, '\n')
    print('resulting pandas dataframe:')
    print(df)
    print('Full Data:')
    print(df.columns)
    print(df.values[:, :])


if __name__ == '__main__':
    main()
