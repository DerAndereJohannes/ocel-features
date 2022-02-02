import pandas as pd
from ocel_features.util.multigraph import create_object_centric_graph


class Object_Situation:
    def __init__(self, log):
        self._log = log
        self._graph = create_object_centric_graph(log)
        self._df = pd.DataFrame({'oid': log['ocel:objects'].keys(),
                                 'type': [log['ocel:objects'][x]['ocel:type']
                                          for x in log['ocel:objects']]})
        self._op_log = []

        # - init function includes code to filter
        #   the objects relative to interest
        # - event -> event, time -> time,
        #   event -> time filtering methods for any/all object types
