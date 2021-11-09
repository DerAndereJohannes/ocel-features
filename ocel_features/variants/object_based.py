import ocel
import numpy as np
from enum import Enum
from ocel_features.util.object_graph import create_object_graph
from ocel_features.util.ocel_alterations import get_activity_names
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA


_FEATURE_PREFIX = 'obj:'


def extract_object_features(log, obj_list=None, feature_list=None):

    net = create_object_graph(log)

    feature_names = list()
    obj_dict = {o: list() for o in obj_list}

    for feature in feature_list:
        curr_feature = getattr(Object_Features, feature.upper(), None)
        if curr_feature is not None:
            curr_feature(feature_names, obj_dict, net, log)
        else:
            print(f'WARNING: Object Feature "{feature}" not found. Skipping.')

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


def extract_activity_existence_lle(feature_list, obj_dict, net, log):
    log_an = get_activity_names(log)
    e_dict = ocel.get_events(log)

    output_components = max(3, int(len(log_an) / 4))
    X_input = np.zeros((len(obj_dict.keys()), len(log_an)), dtype=np.float64)

    an_features = [f'{_FEATURE_PREFIX}activity_embedd:{comp}'
                   for comp in range(output_components)]
    feature_list.extend(an_features)

    for i, o_k in enumerate(obj_dict.keys()):
        obj_an_set = {e_dict[a]['ocel:activity']
                      for a in net.nodes[o_k]['object_events']}

        for j, an in enumerate(log_an):
            if an in obj_an_set:
                X_input[i, j] = np.ushort(1)
            else:
                X_input[i, j] = np.ushort(0)

    an_embedding = LocallyLinearEmbedding(n_components=output_components)
    X_embedded = an_embedding.fit_transform(X_input)

    for i, o_k in enumerate(obj_dict.keys()):
        obj_dict[o_k].extend(X_embedded[i, :])


def extract_activity_existence_pca(feature_list, obj_dict, net, log):
    log_an = get_activity_names(log)
    e_dict = ocel.get_events(log)

    output_components = max(3, int(len(log_an) / 4))
    X_input = np.zeros((len(obj_dict.keys()), len(log_an)), dtype=np.float64)

    an_features = [f'{_FEATURE_PREFIX}activity_embedd:{comp}'
                   for comp in range(output_components)]
    feature_list.extend(an_features)

    for i, o_k in enumerate(obj_dict.keys()):
        obj_an_set = {e_dict[a]['ocel:activity']
                      for a in net.nodes[o_k]['object_events']}

        for j, an in enumerate(log_an):
            if an in obj_an_set:
                X_input[i, j] = np.ushort(1)
            else:
                X_input[i, j] = np.ushort(0)

    an_embedding = PCA(n_components=output_components)
    X_embedded = an_embedding.fit_transform(X_input)

    for i, o_k in enumerate(obj_dict.keys()):
        obj_dict[o_k].extend(X_embedded[i, :])


def extract_object_lifetime(feature_list, obj_dict, net, log):

    e_dict = ocel.get_events(log)
    feature_list.append(f'{_FEATURE_PREFIX}lifetime')

    for o_k, o_v in obj_dict.items():
        obj_time_list = [e_dict[a]['ocel:timestamp']
                         for a in net.nodes[o_k]['object_events']]

        # o_v.append(obj_time_list[-1] - obj_time_list[0])
        if len(obj_time_list) != 0:
            o_v.append(max(0, (max(obj_time_list)
                               - min(obj_time_list)).total_seconds()))
        else:
            o_v.append(0)


def extract_object_unit_set_ratio(feature_list, obj_dict, net, log):
    e_dict = ocel.get_events(log)
    o_dict = ocel.get_objects(log)
    feature_list.append(f'{_FEATURE_PREFIX}single_type_ratio')

    for o_k, o_v in obj_dict.items():
        o_events = net.nodes[o_k]['object_events']
        total_counter = 0
        total_events = len(o_events)
        if total_events != 0:
            for e_k in o_events:
                curr_event = e_dict[e_k]
                obj_same_type = False
                for other_o_k in curr_event['ocel:omap']:
                    if o_k != other_o_k and o_dict[o_k]['ocel:type'] \
                     == o_dict[other_o_k]['ocel:type']:
                        obj_same_type = True
                        break
                if not obj_same_type:
                    total_counter = total_counter + 1

            o_v.append(total_counter / total_events)
        else:
            o_v.append(0)


def extract_avg_object_event_interaction(feature_list, obj_dict, net, log):
    e_dict = ocel.get_events(log)
    feature_list.append(f'{_FEATURE_PREFIX}avg_obj_event_interaction')

    for o_k, o_v in obj_dict.items():
        o_events = net.nodes[o_k]['object_events']
        total_counter = len(o_events) * -1  # subtract self
        total_events = len(o_events)
        if total_events != 0:
            for e_k in o_events:
                curr_event = e_dict[e_k]
                total_counter = total_counter + len(curr_event['ocel:omap'])

            o_v.append(total_counter / total_events)
        else:
            o_v.append(0)


class Object_Features(Enum):
    NEIGHBOUR_COUNT = extract_unique_neighbour_count
    ACTIVITY_EXISTENCE = extract_activity_existence
    _ACTIVITY_EXISTENCE_LLE = extract_activity_existence_lle
    _ACTIVITY_EXISTENCE_PCA = extract_activity_existence_pca
    OBJECT_LIFETIME = extract_object_lifetime
    UNIT_SET_RATIO = extract_object_unit_set_ratio
    AVG_OBJ_INTERACTION = extract_avg_object_event_interaction
