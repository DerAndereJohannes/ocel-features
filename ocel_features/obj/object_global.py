import inspect
import pandas as pd
import numpy as np
import networkx as nx
import numbers
from math import isnan
from collections import Counter
from itertools import product
from ocel_features.util.multigraph import create_object_centric_graph
from ocel_features.util.data_organization import get_basic_stats, \
    Operators, execute_operator, operator_name


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
        col_name = [f'{_FEATURE_PREFIX}type_{ot}_count' for ot in o_types]
        col_values = [np.uint64(0) for _ in o_types]

        # extraction
        o_count = Counter([o['ocel:type']
                           for o in self._log['ocel:objects'].values()])

        for i, col in enumerate(o_types):
            col_values[i] = np.uint64(o_count[col])

        col_name.append(f'{_FEATURE_PREFIX}type_total_count')
        col_values.append(np.uint64(sum(o_count.values())))

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = [col_values]

    def add_global_obj_attribute_stats(self):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        o_types = self._log['ocel:global-log']['ocel:object-types']
        att_types = self._log['ocel:global-log']['ocel:attribute-names']
        col_name = [f'{_FEATURE_PREFIX}']
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

    def add_ot2ot_interactions(self):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        o_types = self._log['ocel:global-log']['ocel:object-types']
        ot2ot = {combo: i for i, combo in enumerate(product(o_types, o_types))}
        col_name = [f'{_FEATURE_PREFIX}obj_{ot1}_to_{ot2}_count'
                    for ot1, ot2 in ot2ot]
        col_values = np.zeros(len(col_name), dtype=np.uint64)
        objects = self._log['ocel:objects']
        # extraction
        for edge in self._graph.edges():
            ot1 = objects[edge[0]]['ocel:type']
            ot2 = objects[edge[1]]['ocel:type']
            otot_index = ot2ot[(ot1, ot2)]
            col_values[otot_index] += 1

            # if graph is undirected, add for the opposite direction
            if isinstance(self._graph, nx.Graph):
                otot_index = ot2ot[(ot2, ot1)]
                col_values[otot_index] += 1

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    # only for directed graphs checking nodes for in and out edges.
    def add_root_leaf_node_count(self):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        leaf_root = ['leaf', 'root']
        obj_types = self._log['ocel:global-log']['ocel:object-types']
        main_index = {combo: i
                      for i, combo in enumerate(product(obj_types, leaf_root))}
        objects = self._log['ocel:objects']
        row_count = len(self._df.index)
        col_name = [f'{_FEATURE_PREFIX}{ot}:{ft}_count'
                    for ot, ft in main_index]

        col_values = np.zeros((row_count, len(col_name)), dtype=np.uint64)

        # extraction
        for node in self._graph.nodes():
            ot = objects[node]['ocel:type']

            if not self._graph.in_edges(node):
                col_values[row_count, main_index[(ot, 'root')]] += 1

            if not self._graph.out_edges(node):
                col_values[row_count, main_index[(ot, 'leaf')]] += 1

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    # only for directed graphs checking nodes for in and out edges.
    def add_separation_complexity_op(self, op=Operators.MAX, absolute=True):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        row_count = len(self._df.index)
        col_name = f'{_FEATURE_PREFIX}{operator_name(op)}:obj_complexity'
        col_values = np.zeros(row_count, dtype=np.float64)

        # extraction
        for i in range(row_count):
            # get root nodes
            curr_nodes = {n for n in self._graph.nodes()
                          if not self._graph.in_degree}

            series = [len(curr_nodes), ]

            # continue until there are no more out_edges
            while curr_nodes:
                new_nodes = set()
                for node in curr_nodes:
                    new_nodes |= self._graph.out_edges(node)

                series.append(len(new_nodes))
                curr_nodes = new_nodes

            # convert to differences
            if not absolute:
                series = [series[i+1] - series[i] for i in range(series[:-1])]

            col_values[i] = execute_operator(op, series)

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
