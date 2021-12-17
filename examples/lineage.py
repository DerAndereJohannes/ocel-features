import ocel
import ocel_features.util.object_descendants as od
from ocel_features.util.convert_neo4j import neo4j_exporter
from dotenv import dotenv_values


config = dotenv_values(".env")
log = ocel.import_log('logs/running-example.jsonocel')

print('creating initial graph')
g = od.create_obj_all_graph(log)
# od.enhance_relations(g)

neo = neo4j_exporter(config['URL'], config['USERNAME'], config['PASSWORD'])

print('importing nodes...')
neo.import_nodes(g)

print('importing relations...')
neo.import_relations(g)

print('Done!')

# [pprint(f'{x} {g.nodes[x]}') for x in g.nodes]
