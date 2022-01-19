from ocel_features.util.multigraph import relations_to_relnames


def obj_relationship_localities(net, rels):
    # setup dicts and make sure no duplicated relations
    rels = relations_to_relnames(rels)

    localities = {}
    for n in net.nodes:
        localities[n] = {rel: set({n}) for rel in rels}

    # follow the edges for every relationship independently
    # improvement possible: do all relationships at the same time
    for obj in localities:
        for rel in rels:
            curr_rel = localities[obj][rel]
            old_neigh = curr_rel
            new_neigh = set()

            while new_neigh or old_neigh is curr_rel:
                new_neigh = set()
                for o in old_neigh:
                    for o2 in net.neighbors(o):
                        if o2 not in curr_rel:
                            if rel in net.edges[o, o2]:
                                new_neigh.add(o2)
                curr_rel = curr_rel | new_neigh
                old_neigh = new_neigh

            localities[obj][rel] = frozenset(curr_rel - {obj})

    return localities


def unique_relationship_localities(localities, rels):
    rel_names = relations_to_relnames(rels)
    uloc = {rel: set() for rel in rel_names}

    for o_rels in localities.values():
        for o_rel in o_rels:
            if o_rel in uloc and o_rels[o_rel]:
                uloc[o_rel].add(o_rels[o_rel])

    return uloc


def unique_relations_to_objects(localities, rels):
    rel_names = relations_to_relnames(rels)
    uloc = {rel: dict() for rel in rel_names}

    for o_id, o_rels in localities.items():
        for o_rel in o_rels:
            if o_rel in uloc and o_rels[o_rel]:
                o_id_set = o_rels[o_rel]
                if o_id_set in uloc[o_rel]:
                    uloc[o_rel][o_id_set].add(o_id)
                else:
                    uloc[o_rel][o_id_set] = {o_id}

    return uloc
