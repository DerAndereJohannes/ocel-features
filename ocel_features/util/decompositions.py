import networkx as nx 


def get_timestamp(log, event):
    return log['ocel:events'][event]['ocel:timestamp']


# returns list double sets. 1st set has connections, 2nd set has connectors, 3rd set has set of object type names used as connectors
def one_many_decomp(log, graph, o_types=None):
    if not o_types:
        o_types = set(log['ocel:object-types'])

    objects = log['ocel:objects']
    events = log['ocel:events']
    return_list = []
    e_d = {}

    for u, v, data in graph.edges(data=True):
        curr_e = next(iter(data['DESCENDANTS']))
        if u not in e_d:
            e_d[u] = ({curr_e}, {v})
        else:
            e_d[u][0].add(curr_e)
            e_d[u][1].add(v)


    for k, v in e_d.items():
        if len(v[0]) > 1:
            pass
        else:
            pass

    print(e_d)


        
    

