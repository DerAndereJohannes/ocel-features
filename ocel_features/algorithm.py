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

    if exe_variant is not None:
        fn, fv = exe_variant(log, entity_list, feature_list)
        df, row_entity_ids = d2p.conv_dict_to_pandas(fn, fv)

    return df, row_entity_ids
