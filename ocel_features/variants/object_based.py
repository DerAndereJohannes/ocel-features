import ocel
from enum import Enum
from ocel_features.util.object_graph import create_object_graph
from ocel_features.util.ocel_alterations import get_activity_names


_FEATURE_PREFIX = 'obj:'


def extract_object_features(log, obj_list=None, feature_list=None):

    if obj_list is None:
        obj_list = ocel.get_objects(log)

    net = create_object_graph(log)

    feature_names = list()
    obj_dict = {o: list() for o in obj_list}

    for feature in feature_list:
        curr_feature = getattr(Object_Features, feature.upper(), None)
        if curr_feature is not None:
            curr_feature(feature_names, obj_dict, net, log)

    return feature_names, obj_dict


def extract_unique_neighbour_count(feature_list, obj_dict, net, log):
    feature_list.append(f'{_FEATURE_PREFIX}neighbour_count')
    for o_k, o_v in obj_dict.items():
        o_v.append(len([x for x in net.neighbors(o_k)]))


def extract_activity_existence(feature_list, obj_dict, net, log):
    log_an = get_activity_names(log)
    e_dict = ocel.get_events(log)

    an_features = [f'{_FEATURE_PREFIX}activity:{an}' for an in log_an]
    feature_list.extend(an_features)

    for o_k, o_v in obj_dict.items():

        obj_an_set = {e_dict[a]['ocel:activity']
                      for a in net.nodes[o_k]['object_events']}

        for an in log_an:
            if an in obj_an_set:
                o_v.append(1)
            else:
                o_v.append(0)


def extract_object_lifetime(feature_list, obj_dict, net, log):

    e_dict = ocel.get_events(log)
    feature_list.append(f'{_FEATURE_PREFIX}lifetime')

    for o_k, o_v in obj_dict.items():
        obj_time_list = [e_dict[a]['ocel:timestamp']
                         for a in net.nodes[o_k]['object_events']]

        # o_v.append(obj_time_list[-1] - obj_time_list[0])
        o_v.append(max(0, (max(obj_time_list)
                           - min(obj_time_list)).total_seconds()))


class Object_Features(Enum):
    NEIGHBOUR_COUNT = extract_unique_neighbour_count
    ACTIVITY_EXISTENCE = extract_activity_existence
    OBJECT_LIFETIME = extract_object_lifetime
