import inspect
import pandas as pd
import numpy as np
import ocel_features.util.ocel_helper as oh
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

    def add_obj_type_counts(self):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        events = self._log['ocel:events']
        objects = self._log['ocel:objects']
        obj_types = self._log['ocel:global-log']['ocel:object-types']
        ot_index = {ot: i for i, ot in enumerate(obj_types)}
        ot_index['TOTAL'] = len(obj_types)
        row_count = len(self._df.index)
        col_name = [f'{_FEATURE_PREFIX}otype_{ot}_count' for ot in obj_types]
        col_name.append(f'{_FEATURE_PREFIX}otype_TOTAL_count')
        col_values = np.zeros((row_count, len(col_name)), dtype=np.uint64)

        for e, vals in events.items():
            event_index = self._ev_index[e]
            for o in vals['ocel:omap']:
                o_type = objects[o]['ocel:type']
                col_values[event_index][ot_index[o_type]] += 1
                col_values[event_index][ot_index['TOTAL']] += 1

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_new_obj_created_counts(self):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        objects = self._graph.nodes()
        obj_types = self._log['ocel:global-log']['ocel:object-types']
        ot_index = {ot: i for i, ot in enumerate(obj_types)}
        ot_index['TOTAL'] = len(obj_types)
        row_count = len(self._df.index)
        col_name = [f'{_FEATURE_PREFIX}obj_{ot}_created_count'
                    for ot in obj_types]
        col_name.append(f'{_FEATURE_PREFIX}otype_TOTAL_created_count')
        col_values = np.zeros((row_count, len(col_name)), dtype=np.uint64)

        for obj, prop in objects.items():
            event_index = self._ev_index[prop['object_events'][0]]
            col_values[event_index][ot_index[prop['type']]] += 1
            col_values[event_index][ot_index['TOTAL']] += 1

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_activity_OHE(self):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        events = self._log['ocel:events']
        activity_names = oh.get_activity_names(self._log)
        an_index = {ot: i for i, ot in enumerate(activity_names)}
        row_count = len(self._df.index)
        col_name = [f'{_FEATURE_PREFIX}activity:{an}'
                    for an in activity_names]
        col_values = np.zeros((row_count, len(col_name)), dtype=np.bool8)

        for e, instance in events.items():
            event_index = self._ev_index[e]
            col_values[event_index][an_index[instance['ocel:activity']]] = 1

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
