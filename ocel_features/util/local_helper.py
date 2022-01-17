def obj_relationship_localities(net, rels):
    # setup dicts and make sure no duplicated relations
    rels = {rel.value[0] for rel in rels}
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


def get_unique_relationship_localities(localities, rels):
    uloc = {rel.value[0]: set() for rel in rels}

    for o_rels in localities.values():
        for o_rel in o_rels:
            if o_rel in uloc and o_rels[o_rel]:
                uloc[o_rel].add(o_rels[o_rel])

    return uloc
