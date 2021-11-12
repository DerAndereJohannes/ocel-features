import inspect
import pandas as pd
import numpy as np
from ocel_features.util.object_graph import create_object_graph
from ocel_features.util.ocel_alterations import get_activity_names


_FEATURE_PREFIX = 'obj:'


# move to other file
def func_name():
    return inspect.stack()[1][3]


class Object_Based:
    def __init__(self, log):
        self._log = log
        self._graph = create_object_graph(log)
        self._df = pd.DataFrame({'oid': log['ocel:objects'].keys(),
                                 'type': [log['ocel:objects'][x]['ocel:type']
                                          for x in log['ocel:objects']]})
        self._op_log = []

    # feature extraction methods
    def add_unique_neighbour_count(self):

        # control
        para_log = (func_name(),)
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        col_name = f'{_FEATURE_PREFIX}neighbour_count'
        row_count = len(self._df.index)
        col_values = np.zeros(row_count, dtype=np.uint64)

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            col_values[i] = len([x for x in self._graph.neighbors(oid)])

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_activity_existence(self):
        # control
        para_log = (func_name(),)
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        log_an = get_activity_names(self._log)
        e_dict = self._log['ocel:events']
        col_name = [f'{_FEATURE_PREFIX}activity:{an}' for an in log_an]
        row_count = len(self._df.index)
        col_values = [np.zeros(row_count, dtype=np.bool8) for an in log_an]

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            obj_an_set = {e_dict[a]['ocel:activity']
                          for a in self._graph.nodes[oid]['object_events']}

            for j in range(len(log_an)):
                if log_an[j] in obj_an_set:
                    col_values[j][i] = 1
                else:
                    col_values[j][i] = 0

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_object_lifetime(self):
        # control
        para_log = (func_name(),)
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        e_dict = self._log['ocel:events']
        col_name = f'{_FEATURE_PREFIX}lifetime'
        row_count = len(self._df.index)
        col_values = np.zeros(row_count, dtype=np.uint64)

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            obj_time_list = [e_dict[a]['ocel:timestamp']
                             for a in self._graph.nodes[oid]['object_events']]

            if len(obj_time_list) != 0:
                col_values[i] = max(0, (max(obj_time_list)
                                    - min(obj_time_list)).total_seconds())

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_obj_unit_set_ratio(self):
        # control
        para_log = (func_name(),)
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        e_dict = self._log['ocel:events']
        o_dict = self._log['ocel:objects']
        col_name = f'{_FEATURE_PREFIX}single_type_ratio'
        row_count = len(self._df.index)
        col_values = np.zeros(row_count, dtype=np.float64)

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            o_events = self._graph.nodes[oid]['object_events']
            total_counter = 0
            total_events = len(o_events)
            if total_events != 0:
                for e_k in o_events:
                    curr_event = e_dict[e_k]
                    obj_same_type = False
                    for other_o_k in curr_event['ocel:omap']:
                        if oid != other_o_k and o_dict[oid]['ocel:type'] \
                         == o_dict[other_o_k]['ocel:type']:
                            obj_same_type = True
                            break
                    if not obj_same_type:
                        total_counter = total_counter + 1

                col_values[i] = total_counter / total_events

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_avg_obj_event_interaction(self):
        # control
        para_log = (func_name(),)
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        e_dict = self._log['ocel:events']
        col_name = f'{_FEATURE_PREFIX}avg_obj_event_interaction'
        row_count = len(self._df.index)
        col_values = np.zeros(row_count, dtype=np.float64)

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            o_events = self._graph.nodes[oid]['object_events']
            total_counter = len(o_events) * -1  # subtract self
            total_events = len(o_events)
            if total_events != 0:
                for e_k in o_events:
                    curr_event = e_dict[e_k]
                    total_counter = total_counter + len(curr_event['ocel:omap'])

                col_values[i] = total_counter / total_events

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
