from numba import types, typed, njit
from numba.experimental import jitclass
import ocel

# THIS CLASS IS ONLY FOR VERY EXPERIMENTAL PURPOSES

obj_spec = [
    ('o_id', types.unicode_type),
    ('first_occurance', types.float64),
    ('first_event', types.unicode_type),
    ('object_events', types.ListType(types.unicode_type)),
    ('type', types.unicode_type)
]


@jitclass(obj_spec)
class OCEL_Object(object):
    def __init__(self, o_id, f_o, f_e, o_e, o_type):
        self.o_id = o_id
        self.first_occurance = f_o
        self.first_event = f_e
        self.object_events = o_e
        self.type = o_type


event_spec = [
    ('e_id', types.unicode_type),
    ('objects', types.ListType(types.unicode_type)),
    ('activity', types.unicode_type),
    ('timestamp', types.float64)
]


@jitclass(event_spec)
class Event(object):
    def __init__(self, e_id, obj_list, activity_name, np_timestamp):
        self.e_id = e_id
        self.objects = obj_list
        self.activity = activity_name
        self.timestamp = np_timestamp


event_dict_types = (types.unicode_type, Event.class_type.instance_type)

log_spec = [
    ('objects', types.ListType(types.unicode_type)),
    ('events', types.DictType(types.unicode_type,
                              Event.class_type.instance_type))
]


@njit
def create_empty_event_list():
    l_new = typed.List([Event('', typed.List(['n']), '', 0.0)])
    l_new.clear()
    return l_new


@njit
def create_empty_event_dict():
    d_new = typed.Dict.empty(*event_dict_types)
    return d_new


@jitclass(log_spec)
class Log(object):
    def __init__(self, obj_list):
        # self.events = typed.Dict.empty(*event_dict_types)
        self.events = create_empty_event_dict()
        self.objects = obj_list


def convert_ocel_to_numba(log):
    jit_log = Log(typed.List(log['ocel:objects'].keys()))

    for e_k, e_v in log['ocel:events'].items():
        jit_log.events[e_k] = Event(e_k, typed.List(e_v['ocel:omap']),
                                    e_v['ocel:activity'],
                                    e_v['ocel:timestamp'].timestamp())

    return jit_log


def main():
    log = ocel.import_log('logs/actual-min.jsonocel')
    return convert_ocel_to_numba(log)


if __name__ == '__main__':
    main()
