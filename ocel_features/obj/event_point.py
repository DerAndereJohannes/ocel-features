import inspect
import pandas as pd
import numpy as np
from ocel_features.util.multigraph import create_object_centric_graph, \
    Relations, _RELATION_DELIMITER

_FEATURE_PREFIX = 'evp:'


# move to other file
def func_name():
    return inspect.stack()[1][3]


class Event_Based:
    def __init__(self, log, graph=None):
        self._log = log
        if graph:
            self._graph = graph
        else:
            self._graph = create_object_centric_graph(log)
        self._df = pd.DataFrame({'eid': log['ocel:events'].keys()})
        self._ev_index = {e: i for i, e in enumerate(log['ocel:events'])}
        self._op_log = []

    def add_relation_created_count(self):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        obj = self._graph.nodes
        rel_names = [r.name for r in Relations
                     if _RELATION_DELIMITER not in r.name]
        rel_names.append('TOTAL')
        rel_index = {r: i for i, r in enumerate(rel_names)}
        row_count = len(self._df.index)
        col_name = [f'{_FEATURE_PREFIX}rel_{r}_count' for r in rel_names]
        col_values = np.zeros((row_count, len(col_name)), dtype=np.uint64)

        # extraction
        for o in obj:
            for o2 in obj:  # for each object combination
                relation_data = self._graph.get_edge_data(o, o2)
                if relation_data:  # if there are relationships
                    for relk, relv in relation_data.items():
                        for e in relv:  # add 1 to each relation in event
                            event_index = self._ev_index[e]
                            col_values[event_index][rel_index[relk]] += 1
                            col_values[event_index][rel_index['TOTAL']] += 1

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    # df methods
    def df_full(self):
        return self._df

    def df_values(self):
        return self._df.select_dtypes(include=np.number).values

    def df_str(self):
        return self._df.select_dtypes(include='O')

    def df_numeric(self):
        return self._df.select_dtypes(include=np.number)

    def get_oid(self, oid):
        return self._df.loc[self._df['oid'] == oid]

    # operator overloads
    def __add__(self, other):
        self._df = pd.concat([self._df, other._df], axis=1)
