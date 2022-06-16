from itertools import chain

from networkx.utils import pairwise, not_implemented_for
import networkx as nx
from copy import deepcopy
import numpy as np

import time

__all__ = ["metric_closure", "steiner_tree"]


@not_implemented_for("directed")
def metric_closure(G, terminal_nodes, weight="weight"):
    """Return the metric closure of a graph.

    The metric closure of a graph *G* is the complete graph in which each edge
    is weighted by the shortest path distance between the nodes in *G* .

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    NetworkX graph
        Metric closure of the graph `G`.

    """
    M = nx.Graph()
    for u in terminal_nodes:
        for v in terminal_nodes:
            length, path = nx.algorithms.shortest_paths.single_source_dijkstra(G, u, v, weight=weight)
            M.add_edge(u, v, distance=length, path=path)
    return M


@not_implemented_for("directed")
def steiner_tree(G, terminal_nodes, weight="weight"):
    """Return an approximation to the minimum Steiner tree of a graph.

    The minimum Steiner tree of `G` w.r.t a set of `terminal_nodes`
    is a tree within `G` that spans those nodes and has minimum size
    (sum of edge weights) among all such trees.

    The minimum Steiner tree can be approximated by computing the minimum
    spanning tree of the subgraph of the metric closure of *G* induced by the
    terminal nodes, where the metric closure of *G* is the complete graph in
    which each edge is weighted by the shortest path distance between the
    nodes in *G* .
    This algorithm produces a tree whose weight is within a (2 - (2 / t))
    factor of the weight of the optimal Steiner tree where *t* is number of
    terminal nodes.

    Parameters
    ----------
    G : NetworkX graph

    terminal_nodes : list
         A list of terminal nodes for which minimum steiner tree is
         to be found.

    Returns
    -------
    NetworkX graph
        Approximation to the minimum steiner tree of `G` induced by
        `terminal_nodes` .

    Notes
    -----
    For multigraphs, the edge between two nodes with minimum weight is the
    edge put into the Steiner tree.


    References
    ----------
    .. [1] Steiner_tree_problem on Wikipedia.
       https://en.wikipedia.org/wiki/Steiner_tree_problem
    """
    # H is the subgraph induced by terminal_nodes in the metric closure M of G.
    M = metric_closure(G, terminal_nodes, weight=weight)
    H = M.subgraph(terminal_nodes)
    # Use the 'distance' attribute of each edge provided by M.
    mst_edges = nx.minimum_spanning_edges(H, weight="distance", data=True)
    # Create an iterator over each edge in each shortest path; repeats are okay
    edges = []
    for u, v, d in mst_edges:
        if d['distance'] == np.inf:
            # check feasibility
            raise ValueError('solution is not feasible.')
        else:
            e = pairwise(d["path"])
            edges.append(e)
    edges = chain.from_iterable(edges)

    # For multigraph we should add the minimal weight edge keys
    if G.is_multigraph():
        edges = (
            (u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for u, v in edges
        )
    T = G.edge_subgraph(edges)
    edges = [e for e in nx.minimum_spanning_edges(T, data=False)]
    T = T.edge_subgraph(edges)
    return T


if __name__ == "__main__":
    import networkx
    from matplotlib import pyplot as plt
    from networkx.algorithms.approximation import steiner_tree as st_origin


    def plot(g, sol, i):
        pos = dict((n, n) for n in g.nodes())
        col = 'black'
        for n in g.nodes:
            alpha = 0.8
            size = 10
            shape = 'o'
            if n in sol.nodes:
                size = 100
            networkx.draw_networkx_nodes(g, pos=pos, nodelist=[n],
                                         node_color=[col], node_shape=shape, alpha=alpha, node_size=size)
        for e in g.edges:
            if e in sol.edges:
                col = 'red'
            else:
                col = 'black'
            networkx.draw_networkx_edges(g, pos=pos, edgelist=[e], edge_color=col)
        plt.title(f"{i}-{len(sol.nodes)}")
        plt.show()

    g = networkx.grid_graph(dim=[10, 10])
    terminals = [(2, 2), (2, 8), (8, 5)]

    sol1 = st_origin(g, terminals)
    plot(g, sol1, 1)

    sol2 = steiner_tree(g, terminals)
    plot(g, sol2, 2)
