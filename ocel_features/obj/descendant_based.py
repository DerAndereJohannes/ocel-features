import ocel
import inspect
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from ocel_features.util.object_graph import create_object_graph
import ocel_features.util.object_descendants as od
from ocel_features.util.ocel_alterations import get_activity_names


_FEATURE_PREFIX = 'desc:'


# move to other file
def func_name():
    return inspect.stack()[1][3]


class Descendant_Based:
    def __init__(self, log):
        self._log = log
        self._desc_graph = od.create_obj_descendant_graph(log)
        self._desc_dict = od.get_obj_descendants(self._desc_graph)
        self._df = pd.DataFrame({'oid': log['ocel:objects'].keys(),
                                 'type': [log['ocel:objects'][x]['ocel:type']
                                          for x in log['ocel:objects']]})
        self._op_log = []

    # Descendant based on single objects
    def add_direct_descendant_count(self):
        # control
        para_log = (func_name(),)
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        col_name = f'{_FEATURE_PREFIX}direct_descendant_count'
        row_count = len(self._df.index)
        col_values = np.zeros(row_count, dtype=np.uint64)

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            col_values[i] = len([x for x in
                                 od.get_direct_descendants(self._desc_graph,
                                                           oid)])

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_total_descendant_count(self):
        para_log = (func_name(),)
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        col_name = f'{_FEATURE_PREFIX}total_descendant_count'
        row_count = len(self._df.index)
        col_values = np.zeros(row_count, dtype=np.uint64)

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            col_values[i] = len(
                self._desc_dict['full_descendants'][oid]['descendants'])

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_obj_type_descendant_count(self):
        para_log = (func_name(),)
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        obj_types = ocel.get_object_types(self._log)
        obj_index = {t: n for n, t in enumerate(obj_types)}
        col_name = [f'{_FEATURE_PREFIX}desc_type_{o_t}' for o_t in obj_types]
        row_count = len(self._df.index)
        col_values = [np.zeros(row_count, dtype=np.uint64) for _ in col_name]

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            counter = [0 for o_t in obj_types]
            for n in self._desc_dict['full_descendants'][oid]['descendants']:
                counter[obj_index[self._desc_graph.nodes[n]['type']]] += 1

            for j, val in enumerate(counter):
                col_values[j, i] = val

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_descendant_relative_ratio(self):
        para_log = (func_name(),)
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        col_name = f'{_FEATURE_PREFIX}descendant_relative_ratio'
        row_count = len(self._df.index)
        col_values = np.zeros(row_count, dtype=np.uint64)

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            desc = len(self._desc_dict['full_descendants'][oid]['descendants'])
            rel = len(self._desc_dict['full_descendants'][oid]['relatives'])
            col_values[i] = desc / (desc + rel)

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
