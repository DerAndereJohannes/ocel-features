from ocel_features.util.log_decompositions import Decompositions, decompose_log


def log_time_decomposition(log, decomp=Decompositions.TIME_DELTA, params=None):
    """Decompse the OCEL log based on time features.

    Args:
        log (Dict): OCEL log from the ocel-standard library.
        decomp (Decompositions, optional): Type of decomposition to apply
        to the log. Defaults to Decompositions.TIME_DELTA.
        params (dict, optional): Any extra parameters requried for the
        execution of the time decomp type. Defaults to None.

    Returns:
        List[Dict]: List of shallow ocel logs decomposed by the user input.
    """
    return decompose_log(log, decomp, params)
