import ocel
from enum import Enum
import ocel_features.util.object_descendants as od


_FEATURE_PREFIX = 'descendant:'


def extract_descendant_features(log, obj_list=None, feature_list=None):

    if obj_list is None:
        obj_list = ocel.get_objects(log)

    data = {}
    output = {}
    data['log'] = log
    data['obj_list'] = obj_list
    data['net'] = od.create_obj_descendant_graph(log)
    data['full_descendants'] = od.get_obj_descendants(data['net'], obj_list)
    output['obj_dict'] = {o: list() for o in obj_list}
    output['feature_names'] = []

    for feature in feature_list:
        curr_feature = getattr(Descendant_Features, feature.upper(), None)
        if curr_feature is not None:
            curr_feature(data, output)
        else:
            print('WARNING:',
                  f'Descendant Feature "{feature}" not found. Skipping.')

    return output['feature_names'], output['obj_dict']


def extract_direct_descendant_count(data, output):
    net = data['net']
    output['feature_names'].append(f'{_FEATURE_PREFIX}direct_descendant_count')
    for o_k, o_v in output['obj_dict'].items():
        o_v.append(len([x for x in od.get_direct_descendants(net, o_k)]))


def extract_total_descendant_count(data, output):
    output['feature_names'].append(f'{_FEATURE_PREFIX}full_descendant_count')
    for o_k, o_v in output['obj_dict'].items():
        o_v.append(len(data['full_descendants'][o_k]['descendants']))


def extract_obj_type_descendant_count(data, output):
    obj_types = ocel.get_object_types(data['log'])
    obj_index = {t: n for n, t in enumerate(obj_types)}
    net = data['net']

    new_features = [f'{_FEATURE_PREFIX}desc_type_{o_t}' for o_t in obj_types]
    output['feature_names'].extend(new_features)

    for o_k, o_v in output['obj_dict'].items():
        counter = [0 for o_t in obj_types]
        for node in data['full_descendants'][o_k]['descendants']:
            counter[obj_index[net.nodes[node]['type']]] += 1
        o_v.extend(counter)


def extract_descendant_relative_ratio(data, output):
    output['feature_names'].append(
        f'{_FEATURE_PREFIX}descendant_relative_ratio')

    for o_k, o_v in output['obj_dict'].items():
        descendants = data['full_descendants'][o_k]['descendants']
        relatives = data['full_descendants'][o_k]['relatives']

        o_v.append(len(descendants) / (len(descendants) + len(relatives)))


class Descendant_Features(Enum):
    DIRECT_DESCENDANT_COUNT = extract_direct_descendant_count
    TOTAL_DESCENDANT_COUNT = extract_total_descendant_count
    DESCENDANT_TYPE_COUNT = extract_obj_type_descendant_count
    DESCENDANT_RELATIVE_RATIO = extract_descendant_relative_ratio
