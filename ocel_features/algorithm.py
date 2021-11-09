from enum import Enum
import ocel_features.variants.object_based as ob
import ocel_features.variants.descendant_based as db
import ocel_features.util.conv_dict_pandas as d2p
import ocel_features.util.ocel_alterations as oa


class Variants(Enum):
    DEFAULT = ob.extract_object_features
    OBJECT_BASED = ob.extract_object_features
    DESCENDANT_BASED = db.extract_descendant_features


def apply(log, variant='default', entity_list=None, feature_list=None):
    exe_variant = getattr(Variants, variant.upper(), None)
    oa.remove_empty_entities(log)
    df, row_entity_ids = None, []

    # Processing the entity_list input
    if isinstance(entity_list, str) \
       and entity_list in log['ocel:global-log']['ocel:object-types']:
        obj_type = entity_list
        entity_list = [o_k for o_k, o_v in log['ocel:objects'].items()
                       if o_v['ocel:type'] == obj_type]
    elif entity_list is None:
        entity_list = log['ocel:objects']
    elif not isinstance(entity_list, list):
        raise InvalidObjectTypeException(
            f'"{entity_list}" is not an object type in this OCEL-Log.')

    # Processing the feature_list input
    if feature_list is None:
        feature_list = [f for f in vars(ob.Object_Features) if f[0] != '_']

    if exe_variant is not None:
        fn, fv = exe_variant(log, entity_list, feature_list)
        df, row_entity_ids = d2p.conv_dict_to_pandas(fn, fv)

    return df, row_entity_ids


class InvalidObjectTypeException(Exception):
    pass
