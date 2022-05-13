import inspect
import pandas as pd
import numpy as np
from itertools import product
from collections import Counter
from ocel_features.util.ocel_helper import get_activity_names
from ocel_features.util.experimental.convert_neo4j \
    import event_directly_follows
from ocel_features.util.data_organization import Operators, \
    execute_operator, operator_name


_FEATURE_PREFIX = 'evg:'


# move to other file
def func_name():
    return inspect.stack()[1][3]


class Event_Global:
    def __init__(self, log, graph=None):
        self._log = log
        if graph:
            self._graph = graph
        else:
            self._graph = event_directly_follows(log)
        self._df = pd.DataFrame()
        self._misc = {}
        self._op_log = []

    def add_activity_counts(self):
        # control
        para_log = (func_name(), )
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        row_count = len(self._df.index)
        events = self._log['ocel:events']
        an_index = {an: i
                    for i, an in enumerate(get_activity_names(self._log))}

        col_name = [f'{_FEATURE_PREFIX}{an}_count'
                    for an in an_index]

        col_values = np.zeros((row_count, len(col_name)), dtype=np.uint64)

        # extraction
        for i in range(row_count):
            for en, ev in events.items():
                activity = ev['ocel:activity']
                col_values[i][an_index[activity]] += 1

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_activity_value_operator(self, op=Operators.SUM):
        # control
        para_log = (func_name(),)
        if para_log in self._op_log:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        log_an = get_activity_names(self._log)
        log_pn = self._log['ocel:global-log']['ocel:attribute-names']

        # save the column number
        events = self._log['ocel:events']
        anpn_combo = {combo: i
                      for i, combo in enumerate(product(log_an, log_pn))}

        col_name = [f'{_FEATURE_PREFIX}activity:{an}:{pn}'
                    for an, pn in anpn_combo]
        row_count = len(self._df.index)
        col_values = [np.zeros(row_count, dtype=np.float64)
                      for _ in col_name]

        # extraction
        for i in range(row_count):
            e_values = {anpn: [] for anpn in anpn_combo}

            for e in events:
                an = e['ocel:activity']
                for pn, value in e['ocel:vmap']:
                    e_values[(an, pn)].append(value)

            for j, anpn in enumerate(anpn_combo):
                col_values[j][i] = execute_operator(op, e_values[anpn])

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_activity_ot_operator(self, op=Operators.SUM):
        # control
        para_log = (func_name(),)
        if para_log in self._op_log:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        log_an = get_activity_names(self._log)
        log_ot = self._log['ocel:global-log']['ocel:object-types']
        op_name = operator_name(op)

        # save the column number
        events = self._log['ocel:events']
        objects = self._log['ocel:objects']
        anot_combo = {combo: i
                      for i, combo in enumerate(product(log_an, log_ot))}

        col_name = [f'{_FEATURE_PREFIX}activity:{an}:{ot}_count_{op_name}'
                    for an, ot in anot_combo]
        row_count = len(self._df.index)
        col_values = [np.zeros(row_count, dtype=np.float64)
                      for _ in col_name]

        # extraction
        for i in range(row_count):
            e_values = {anot: [] for anot in anot_combo}

            for e in events.values():
                an = e['ocel:activity']
                ot_counter = Counter([objects[o]['ocel:type']
                                      for o in e['ocel:omap']])
                for ot, value in ot_counter:
                    e_values[(an, ot)].append(value)

            for j, anot in enumerate(anot_combo):
                col_values[j][i] = execute_operator(op, e_values[anot])

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_activity_df_time_operator(self, op=Operators.SUM):
        # control
        para_log = (func_name(),)
        if para_log in self._op_log:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        log_an = get_activity_names(self._log)
        op_name = operator_name(op)

        # save the column number
        events = self._log['ocel:events']
        anot_combo = {combo: i
                      for i, combo in enumerate(product(log_an, log_an))}

        col_name = [f'{_FEATURE_PREFIX}activity:{an1}-{an2}:time_{op_name}'
                    for an1, an2 in anot_combo]
        row_count = len(self._df.index)
        col_values = [np.zeros(row_count, dtype=np.float64)
                      for _ in col_name]

        # extraction
        for i in range(row_count):
            e_values = {anot: [] for anot in anot_combo}

            for eid, e in events.items():
                an1 = e['ocel:activity']
                t1 = e['ocel:timestamp']

                out_edges = self._graph.out_edges(eid)
                for oe in out_edges:
                    an2 = events[oe]['ocel:activity']
                    t2 = events[oe]['ocel:timestamp']
                    e_values[(an1, an2)].append((t2 - t1).total_seconds())

            for j, anot in enumerate(anot_combo):
                col_values[j][i] = execute_operator(op, e_values[anot])

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
        col_name = f'{_FEATURE_PREFIX}{operator_name(op)}:event_complexity'
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
