import pandas as pd
import numpy as np
from ocel_features.util.object_graph import create_object_graph


class Object_Based:
    def __init__(self, log):
        self._log = log
        self._object_graph = create_object_graph(log)
        self._df = pd.DataFrame({'oid': log['ocel:objects'].keys(),
                                 'type': [log['ocel:objects'][x]['ocel:type']
                                          for x in log['ocel:objects']]})

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
