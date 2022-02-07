import inspect
import pandas as pd
import numpy as np
import numbers
from math import isnan
from collections import Counter
from ocel_features.util.multigraph import create_object_centric_graph
from ocel_features.util.data_organization import get_basic_stats


_FEATURE_PREFIX = 'objg:'


# move to other file
def func_name():
    return inspect.stack()[1][3]


class Object_Global:
    def __init__(self, log, graph=None):
        self._log = log
        if graph:
            self._graph = graph
        else:
            self._graph = create_object_centric_graph(log)
        self._df = pd.DataFrame()
        self._misc = {}
        self._op_log = []

    def add_global_obj_type_count(self):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        o_types = self._log['ocel:global-log']['ocel:object-types']
        col_name = [f'{_FEATURE_PREFIX}:type_{ot}_count' for ot in o_types]
        col_values = [np.uint64(0) for _ in o_types]

        # extraction
        o_count = Counter([o['ocel:type'] for o in self._log['ocel:objects']])
        for i, col in enumerate(o_types):
            col_values[i] = np.uint64(o_count[col])

        col_name.append(f'{_FEATURE_PREFIX}:type_total_count')
        col_values.append(np.uint64(sum(o_count.values())))

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_global_obj_attribute_stats(self):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        o_types = self._log['ocel:global-log']['ocel:object-types']
        att_types = self._log['ocel:global-log']['ocel:attribute-names']
        col_name = []
        col_values = []
        holder_dict = {ot: {att: [] for att in att_types} for ot in o_types}

        # extraction
        for o in self._log['ocel:objects'].values():
            o_type = o['ocel:type']
            for att, val in o['ocel:ovmap']:
                if val is not None and (isinstance(val, numbers.Number)
                   and not isnan(val)):
                    holder_dict[o_type][att].append(val)

        for ot, atts in holder_dict.items():
            col_name.append(f'{_FEATURE_PREFIX}:type_{ot}_count')
            col_values.append(np.uint64(len(atts)))
            for att in atts:
                new_cols, new_vals = get_basic_stats(
                    f'{_FEATURE_PREFIX}:type_{ot}_', att)
                col_name.extend(new_cols)
                col_values.extend(new_vals)

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
