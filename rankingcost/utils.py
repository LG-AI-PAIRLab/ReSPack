import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from itertools import combinations


def get_twopin_list(pinList):
    # node set -> two pin list
    # run mst on given nodes (not original graph we want to route)

    # Set Manhattan distance as weights of tree
    recDist = np.zeros((len(pinList),len(pinList)))
    for i in range(len(pinList)):
        for j in range(len(pinList)):
            recDist[i,j] = np.abs(np.array(pinList[i]) - np.array(pinList[j])).sum()

    X = csr_matrix(recDist)
    Tcsr = minimum_spanning_tree(X)
    Tree = Tcsr.toarray().astype(int)
    
    twoPinListSorted = []
    for i in range(Tree.shape[0]):
        for j in range(Tree.shape[1]):
            if Tree[i,j] != 0:
                twoPinListSorted.append([pinList[i],pinList[j]])
    return twoPinListSorted

def extract_subgraph(solver):
    ST = {}
    edge_solutions, order = solver.best_traces, solver.best_solution
    if edge_solutions is None:
        return None

    for t in order:
        edges, nodes = [], []
        for i, j in edge_solutions[t]:
            edges.append((i, j))
            nodes.append(i)
            nodes.append(j)
        ST[t] = (edges, list(set(nodes)))
    return ST


def check_spacing(G, prev_margins, new_nodes, n_space):
    """
    check spacing constraints and return flag and margin node set.
    Parameters
    -------------
    prev_margins    : list of set.  
    new_nodes       : list of nodes
    n_space         : scalar.

    Returns
    -------------
    is_violated     : boolean.
    info            : set of nodes. it means nodes <= n_space
    """
    from functools import reduce
    violated = []
    # check whether new path violate previous margin nodes
    prev_all_margins = list(reduce(lambda x, y: x|y, prev_margins, set()))
    for nn in new_nodes:
        if nn in prev_all_margins:
            violated.append(nn)

    def get_margins(nodes):
        margins = nodes.copy()
        for n in nodes:
            for nei in G.neighbors(n):
                margins.add(nei)
        return margins

    new_margins = set(new_nodes)
    for _ in range(n_space):
        new_margins = get_margins(new_margins)
    new_margins -= set(new_nodes)

    is_violated, info = False if not violated else True, new_margins
    return is_violated, info

def check_radius(G, new_edges, terminals, radius):
    is_violated, max_radius = False, -1
    
    if radius <= 0: 
        return is_violated, max_radius  # False, -1

    import networkx as nx
    nx.set_edge_attributes(G, float('inf'), 'routed')

    for ed in new_edges:
        G.edges[ed]['routed'] = 1 

    # true max length
    for t1, t2 in combinations(terminals, 2):
        r = nx.shortest_path_length(G, t1, t2, 'routed')
        if max_radius < r:
            max_radius = r

    # radius constraints (max_radius <= radius )
    is_violated, info = radius < max_radius, max_radius
    return is_violated, info

def is_violate_constraints(G, prev_margins, new_st, terminals_a_net, const_a_net):
    new_edges, new_nodes = new_st
    # no constraints
    if ('constraints' not in G.graph.keys()) or (not G.graph['constraints']): 
        return False, (-1, set())
    # const = G.graph['constraints'][net_id]
    result_r, info_r = check_radius(G, new_edges, terminals_a_net, const_a_net['radius'])
    result_s, info_s = check_spacing(G, prev_margins, new_nodes, const_a_net['margin'])

    is_violated =  result_r or result_s
    return is_violated, (info_r, info_s)