from enum import Enum
from ocel_features.obj.object_point import Object_Based
from ocel_features.obj.object_global import Object_Global
from ocel_features.obj.event_point import Event_Based
# from ocel_features.obj.event_global import Event_Global


# Object based
class OBJECT_FEATURE_TYPE(Enum):
    POINT = (Object_Based,)
    GLOBAL = (Object_Global,)


def object_features(log, graph=None, f_type=OBJECT_FEATURE_TYPE.POINT):
    """Create a pandas dataframe object consisting of objects from the OCEL
    log based on the target feature type the user is trying to analyse. With
    the resulting object, the user can apply various feature extraction methods
    on each of the targets.

    Args:
        log (Dict): Object-centric event log
        graph (DiGraph, optional): Object-centric directed graph.
        Defaults to None.
        f_type (OBJECT_FEATURE_TYPE, optional): Type of features that should be
        analysed. Defaults to OBJECT_FEATURE_TYPE.POINT.

    Returns:
        Object Table Object: OCEL object dataframe special object with feature
        extraction methods
    """
    object_feature_type, = f_type.value

    return object_feature_type(log, graph)


# Event based
class EVENT_FEATURE_TYPE(Enum):
    POINT = (Event_Based,)
#     GLOBAL = (Event_Global,)


def event_features(log, graph=None, f_type=EVENT_FEATURE_TYPE.POINT):
    """Create a pandas dataframe object consisting of events from the OCEL
    log based on the target feature type the user is trying to analyse. With
    the resulting object, the user can apply various feature extraction methods
    on each of the targets.

    Args:
        log (Dict): Object-centric event log
        graph (DiGraph, optional): Object-centric directed graph.
        Defaults to None.
        f_type (EVENT_FEATURE_TYPE, optional): Type of features that should be
        analysed. Defaults to EVENT_FEATURE_TYPE.POINT.

    Returns:
        Event Table Object: OCEL event dataframe special object with feature
        extraction methods
    """
    event_feature_type, = f_type.value
    return event_feature_type(log, graph)
