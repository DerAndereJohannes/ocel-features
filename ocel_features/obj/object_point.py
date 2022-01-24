import inspect
import pandas as pd
import numpy as np
import ocel_features.util.relations_helper as rh
from sklearn.decomposition import PCA
from ocel_features.util.multigraph import create_object_centric_graph
from ocel_features.util.ocel_helper import get_activity_names


_FEATURE_PREFIX = 'objp:'


# move to other file
def func_name():
    return inspect.stack()[1][3]


class Object_Based:
    def __init__(self, log):
        self._log = log
        self._graph = create_object_centric_graph(log)
        self._df = pd.DataFrame({'oid': log['ocel:objects'].keys(),
                                 'type': [log['ocel:objects'][x]['ocel:type']
                                          for x in log['ocel:objects']]})
        self._op_log = []

    # feature extraction methods DONE
    def add_unique_neighbour_count(self):
        # control
        para_log = (func_name(),)
        if para_log in self._op_log:
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
        if para_log in self._op_log:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        log_an = get_activity_names(self._log)
        e_dict = self._log['ocel:events']
        col_name = [f'{_FEATURE_PREFIX}activity:{an}' for an in log_an]
        row_count = len(self._df.index)
        col_values = [np.zeros(row_count, dtype=np.bool8) for _ in log_an]

        # extraction DONE
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            obj_an_set = {e_dict[a]['ocel:activity']
                          for a in self._graph.nodes[oid]['object_events']}

            for j in range(len(log_an)):
                if log_an[j] in obj_an_set:
                    col_values[j][i] = 1

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_activity_existence_pca(self, n_components=None):
        # parameter checking
        log_an = get_activity_names(self._log)
        if not isinstance(n_components, int):
            n_components = max(3, int(len(log_an) / 4))

        # control
        para_log = (func_name(), n_components)
        if para_log in self._op_log:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        e_dict = self._log['ocel:events']
        col_name = [f'{_FEATURE_PREFIX}activity_embedd_pca:{comp}'
                    for comp in range(n_components)]
        row_count = len(self._df.index)
        x_input = np.zeros((row_count, len(log_an)), dtype=np.bool8)

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            obj_an_set = {e_dict[a]['ocel:activity']
                          for a in self._graph.nodes[oid]['object_events']}

            for j in range(len(log_an)):
                if log_an[j] in obj_an_set:
                    x_input[i, j] = 1

        an_embedding = PCA(n_components=n_components)
        col_values = an_embedding.fit_transform(x_input)

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_object_lifetime(self):
        # control
        para_log = (func_name(),)
        if para_log in self._op_log:
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
        if para_log in self._op_log:
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
            tot_counter = 0
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
                        tot_counter = tot_counter + 1

                col_values[i] = tot_counter / total_events

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_avg_obj_event_interaction(self):
        # control
        para_log = (func_name(),)
        if para_log in self._op_log:
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
            tot_counter = len(o_events) * -1  # subtract self
            total_events = len(o_events)
            if total_events != 0:
                for e_k in o_events:
                    curr_event = e_dict[e_k]
                    tot_counter = tot_counter + len(curr_event['ocel:omap'])

                col_values[i] = tot_counter / total_events

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_obj_type_interaction(self, obj_types=None):
        # control
        para_log = (func_name(), obj_types)
        if para_log in self._op_log:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        o_dict = self._log['ocel:objects']
        all_types = self._log['ocel:global-log']['ocel:object-types']

        if obj_types is None:
            obj_types = all_types
        else:
            obj_types = [o_type for o_type in obj_types if o_type in all_types]

        o_types = {o_type: i for i, o_type in
                   enumerate(obj_types)}

        col_name = [f'{_FEATURE_PREFIX}interaction_with:{ot}'
                    for ot in o_types]
        row_count = len(self._df.index)
        col_values = [np.zeros(row_count, dtype=np.uint64) for _ in o_types]

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]

            for neigh in self._graph.neighbors(oid):
                col_num = o_types[o_dict[neigh]['ocel:type']]
                col_values[col_num][i] += 1

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_object_dfollows(self):
        # control
        para_log = (func_name(),)
        if para_log in self._op_log:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        log_an = get_activity_names(self._log)
        an_product = [(an1, an2) for an1 in log_an
                      for an2 in log_an if an1 != an2]
        an_index = {anp: i for i, anp in enumerate(an_product)}

        e_dict = self._log['ocel:events']
        col_name = [f'{_FEATURE_PREFIX}df:{df[0]}:{df[1]}'
                    for df in an_product]
        row_count = len(self._df.index)
        col_values = [np.zeros(row_count, dtype=np.uint64) for _ in an_product]

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            obj_an = [e_dict[a]['ocel:activity']
                      for a in self._graph.nodes[oid]['object_events']]

            for a_i in range(len(obj_an[:-1])):
                df_rel = (obj_an[a_i], obj_an[a_i + 1])
                col_values[an_index[df_rel]][oid] += 1

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    def add_obj_wait_time(self, source, target):
        # control
        para_log = (func_name(), source, target)
        if para_log in self._op_log:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        log_an = get_activity_names(self._log)
        if source not in log_an or target not in log_an:
            print(f'[!] {source} -> {target} is invalid (check names).')
            return
        e_dict = self._log['ocel:events']
        col_name = f'{_FEATURE_PREFIX}{source}:{target}:wait_time'
        row_count = len(self._df.index)
        col_values = np.zeros(row_count, dtype=np.uint64)

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            obj_events = self._graph.nodes[oid]['object_events']
            obj_an = {e_dict[a]['ocel:activity'] for a in obj_events}

            if not (source not in obj_an or target not in obj_an):
                src_time = None
                tar_time = None
                # takes the time of the first instance of src -> tar
                for event in obj_events:
                    if event['ocel:activity'] == target:
                        if not src_time:
                            tar_time = event['ocel:timestamp']
                    elif event['ocel:activity'] == source:
                        if not src_time:
                            src_time = event['ocel:timestamp']

                    if src_time and tar_time:
                        break

                # making sure there are values
                if not src_time and not tar_time:
                    result = (tar_time - src_time).total_seconds()

                    if result > 0:
                        col_values[i] = result

        # add to df
        self._op_log.append(para_log)
        self._df[col_name] = col_values

    # Descendant based on single objects
    def add_direct_rel_count(self, rels=['descendants', 'ancestors']):
        # control
        para_log = (func_name(), rels)
        if para_log in self._df.columns:
            print(f'[!] {para_log} already computed. Skipping..')
            return

        # df setup
        col_name = [f'{_FEATURE_PREFIX}df:direct_{rel}_count' for rel in rels]
        row_count = len(self._df.index)
        col_values = [np.zeros(row_count, dtype=np.uint64) for _ in rels]

        # extraction
        for i in range(row_count):
            oid = self._df.iloc[i, 0]
            relations = rh.get_direct_relations_count(self._graph, oid)

            for j, rel in enumerate(rels):
                col_values[i, j] = relations[rel]

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
