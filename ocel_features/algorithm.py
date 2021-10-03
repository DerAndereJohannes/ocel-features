from enum import Enum
import ocel_features.variants.object_based as ob
import ocel_features.util.conv_dict_pandas as d2p


class Variants(Enum):
    DEFAULT = ob.extract_object_features
    OBJECT_BASED = ob.extract_object_features


def apply(log, variant='default', entity_list=None, feature_list=None):
    exe_variant = getattr(Variants, variant.upper(), None)
    df = None
    if exe_variant is not None:
        fn, fv = exe_variant(log, entity_list, feature_list)
        df, row_entity_ids = d2p.conv_dict_to_pandas(fn, fv)

    return df, row_entity_ids
