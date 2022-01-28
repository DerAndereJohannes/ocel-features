import ocel
from ocel_features.obj.event_point import Event_Based


def main():
    a = Event_Based(ocel.import_log('logs/actual-min.jsonocel'))
    # a.add_relation_created_count()
    # a.add_obj_type_counts()
    # a.add_new_obj_created_counts()
    a.add_activity_OHE()
    print(a.df_full().columns)
    print(a.df_full().values)


if __name__ == '__main__':
    main()
