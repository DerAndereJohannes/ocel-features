from ocel_features.util.log_series import LogFunctions, apply_func_to_binning,\
    series_differences_absolute, series_differences_percentage
# from ocel_features.util.graph_series import


def log_decomp_series(ordered_sublogs,
                      func=LogFunctions.ACTIVITY_COUNT,
                      params=None):
    """Create a simple series based on all of the sublogs contained in a list.
    Each sublog is represented by one value calculated by the desired input
    function.

    Args:
        ordered_sublogs (List[Dict]): List of ordered sublogs.
        func (LogFunctions, optional): The desired input function to base the
        series off of. Defaults to LogFunctions.ACTIVITY_COUNT.
        params (Dict, optional): Parameters required for the function to work
        (if required). Defaults to None.

    Returns:
        List[Number]: Series based on the input sublogs, function
        and parameters.
    """
    return apply_func_to_binning(ordered_sublogs, func, params)


def convert_series_absolute_differences(series):
    """Convert series list to series of absolute differences to the previous
    value in the series.

    Args:
        series (List[Number]): Ordered list of values

    Returns:
        List[Number]: List of numbers based on their absolute difference
        to the previous value.
    """
    return series_differences_absolute(series)


def convert_series_relative_differences(series, norm=False):
    """Convert series list to series of relative differences to the previous
    value in the series.

    Args:
        series (List[Number]): Ordered list of values.
        norm (bool, optional): Decide if the list should be normed to the max
        value. Defaults to False.

    Returns:
        List[Number]: List of numbers based on their relative difference to
        the previous value.
    """
    return series_differences_percentage(series, norm)
