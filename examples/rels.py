import networkx as nx

def main():
    g = nx.DiGraph()
    g.add_edge(1,2, d=True)
    g.add_edge(2,1, a=True)
    g.add_edge(2,3, cb=True)
    g.add_edge(3,2, cb=True)

    return g
