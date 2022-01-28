import ocel
from ocel_features.obj.object_point import Object_Based


def main():
    a = Object_Based(ocel.import_log('logs/actual-min.jsonocel'))
    # a.add_direct_rel_count()
    a.add_obj_start_end()
    print(a.df_full().columns)
    print(a.df_full().values)


if __name__ == '__main__':
    main()
