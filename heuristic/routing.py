import heapq as hq
from itertools import chain
from heuristic.utils import init_startnodes, init_visited, backtrace
from common.graph import get_complete, find_mst, restore_graph, remove_inessentials


def Lee(graph, net_id, c):
    '''
    Lee's algorithm which is one of 2-points based heuristic routing algorithms.
    
    Parameters
    -------------
    graph : class. e.g., GridGraph()
    net_id : tuple. key of a single net
    c : dict. cost of nodes

    Returns
    -------------
    st_edges : list of tuples. e.g., [ [(0,0,0), (0,1,0)], [(0,1,0), (1,1,0)] ]
    st_nodes : list of nodes. e.g., [ (0,0,0), (0,1,0), (1,1,0) ]

    References
    -------------
    Lee, Chin Yang. "An algorithm for path connections and its applications." IRE transactions on electronic computers 3 (1961): 346-365.
    Takahashi and Matsuyama. An approximate solution for the problem in graphs". Japonica (1980):573-577
    '''
    
    terminals = graph.netlists[net_id]
    st_edges = set()

    # get start node
    starts, lefts = init_startnodes(terminals[:])
    visited = init_visited(graph.nodes, starts)
    nodes = [(0, [starts])]  # length, history

    hq.heapify(nodes)
    while nodes and lefts:  # if there are no nodes or terminals to visit, then it stops
        length, history = hq.heappop(nodes) # pop short-length first
        target = history[-1]
        # 1) visit terminal
        if target in lefts: 
            st_edges.update(backtrace(history))
            lefts.remove(target)
            nodes = [ (0, [n]) for n in history ]   # all nodes of history will be starting points
            hq.heapify(nodes)
            visited = init_visited(graph.nodes, history)    # initialize 'visited'
            continue
        # 2) not terminal
        for nei in graph.neighbors(target): # add neighbor
            # current cost + edge weight + neighbor cost (weight=1)
            if (length + graph.edges[target, nei]['weight'] + c[nei]) >= visited[nei]: 
                continue
            visited[nei] = length + graph.edges[target, nei]['weight'] + c[nei]
            hq.heappush(
                nodes,
                (length + graph.edges[target, nei]['weight'] + c[nei], history + [nei])
            )
    if lefts:   # fail to route all terminals
        st_nodes = []
    else:
        # unique node set
        st_nodes = list(set(chain(*st_edges)))  # st_edges=[(s1, s2), (s2, s3), ..., (s(n-1), s(n))], st_nodes = [s1,s2, ..., s(n-1), s(n)]
    return list(st_edges), st_nodes


def TwoApprox(graph, net_id, c):
    '''
    find a single Steiner tree.

    Step 1: compute the complete undirected subgraph G1 of terminal nodes in original graph G, where the complete graph is fully-connected graph in which each edge is weighted by the shortest path distance between the nodes.
    Step 2: find the minimum spanning tree T1 of G1. (remove redundant edge of virtual graph which consists of terminal nodes)
    Step 3: construct the subgraph Gs which replace each edge in T1 by shortest path of G.
    Step 4: find the minimum spanning tree Ts of Gs. (remove cyclic edges of Gs)
    Step 5: make sure all the leaves in Ts are terminal nodes by deleting edges in Ts, if necessary.

    References
    ------------    
    Kou, Lawrence, George Markowsky, and Leonard Berman. "A fast algorithm for Steiner trees." Acta informatica 15.2 (1981): 141-145.
    '''
    graph.update_edge_cost(c) # make 'cost' attribute by using node and edge weight
    terminals = graph.netlists[net_id]

    terminal_complete = get_complete(graph, terminals)
    terminal_mst = find_mst(terminal_complete, 'weight')
    cand_subgraph = restore_graph(graph, terminal_mst)
    subgraph = find_mst(cand_subgraph, 'cost')
    ST = remove_inessentials(subgraph, terminals)

    return list(ST.edges), list(ST.nodes)
