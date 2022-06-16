def make_roots(tset, n_trials=10):
    """
    make all combinations of terminal sets
    :param tset : terminal sets <list>
    :param n_trials : # of sampled root sets <int>
    e.g., terminal set = [(1,2), (3,4), (1,2)]
                     R = [(1,3,1), (1,3,2), (1,4,1), (1,4,2) ...]
    """

    import random
    from math import prod

    n_trials = min(n_trials, prod([len(t) for t in tset]))
    selected_roots = []
    while len(selected_roots) < n_trials:
        T = tuple([random.choice(t) for t in tset])
        if T not in selected_roots:
            selected_roots.append(T)
    return selected_roots


def t_generator(tsets):
    """
    a generator for visiting all terminal nodes
    :param tsets : terminal sets <list> (e.g., [(1,2,3), (4,5)])
    :return n : index of terminal set <int>
    :return idx_t : index of terminal node <int>
    :return t : terminal node <int>
    """
    idx_t = 0
    for n in range(len(tsets)):
        for t in tsets[n]:
            yield n, idx_t, t
            idx_t += 1


def extract_subgraph(X):
    """
    find connected edge(x=1) and node sets from X
    :param X :
    """
    ST = []
    for x in X:
        edges, nodes = [], []
        for i, j in x.keys():
            if x[(i, j)].solution_value() == 1:
                edges.append((i, j))
                nodes.append(i)
                nodes.append(j)
        ST.append((edges, list(set(nodes))))
    return ST
