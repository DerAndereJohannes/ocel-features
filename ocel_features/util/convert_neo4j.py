from neo4j import GraphDatabase


class neo4j_exporter:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_nodes(self, graph):
        exe_list = ["CREATE "]

        for n in graph.nodes():
            exe_list.append(f"(:Object:{graph.nodes[n]['type']}"
                            f" {create_node_str(graph.nodes[n])}),")

        exe_list[-1] = exe_list[-1][:-1]  # remove last comma

        import_nodes_query = ''.join(exe_list)
        with self.driver.session() as session:
            session.run(import_nodes_query)

    def import_relations(self, graph):
        exe_list = []

        for e in graph.edges():
            src = e[0]
            tar = e[1]
            rel = graph.edges[e]['rel']
            exe_list.append(f"MATCH (a:Object), (b:Object) "
                            f"WHERE a.id='{src}' AND b.id='{tar}' "
                            f"CREATE (a)-[:{rel}]->(b)")
        # print(exe_list)
        with self.driver.session() as session:
            for q in exe_list:
                session.run(q)


def create_node_str(n_dict):
    return f"{{id: '{n_dict['id']}'," \
                f"first_occurance: {n_dict['first_occurance']}}}"
