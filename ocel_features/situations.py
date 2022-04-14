from ocel_features.obj.object_situation import create_situations, Targets

DEFAULT_TARGET = Targets.EVENTCHOICE


def extract_situations(log, graph, targets, target_feature=DEFAULT_TARGET,
                       params=None):
    """Extract situations based on the user's input. A situation is an empty
    table of features with the target feature based on the user's input. Eg.
    Target feature being that the next activity is X. Based on the situation
    information, extracted features can then be used to act as test data for
    the target value.

    Args:
        log (Dict): Object-centric event log
        graph (DiGraph): Object-centric directed graph
        targets (List[String]): List of objects to test if they can be a target
        based on the requirements.
        target_feature (Targets, optional): Type of situation that the user
        would like to analyse. Defaults to DEFAULT_TARGET.
        params (Dict, optional): Parameters required for the target_feature if
        required. Defaults to None.

    Returns:
        List[Situation]: List of situations that can be analysed
        based on the user's input.
    """
    return create_situations(log, graph, targets, target_feature, params)
