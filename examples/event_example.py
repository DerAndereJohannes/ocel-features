import ocel
from ocel_features.obj.event_point import Event_Based


def main():
    a = Event_Based(ocel.import_log('logs/actual-min.jsonocel'))
    a.add_relation_created_count()

    print(a.df_full().columns)
    print(a.df_values())


if __name__ == '__main__':
    main()
