import os
import re
import math
from itertools import chain, combinations
from collections import Counter, defaultdict
from collections.abc import Iterable
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GridGraph(nx.Graph):
    def __init__(self, n=None, m=None, g=None, sol=None, idx_netlists=None):
        super().__init__()
        if n or m or g or idx_netlists:
            self._init_graph(n, m, idx_netlists, g)
        self.solution = sol

    def _get_idx_from_coord(self, coord):
        x, y, z = coord
        return y * self.m + x

    def _get_coord_from_idx(self, idx):
        return (idx % self.m, idx // self.m, 0)

    def _get_2d_gridgraph(self, n, m):
        """
        generate nodes and edge in NXM grid.
        every coordinate is 3d (x, y, z=0).
        all edges are in ascending order. (e.g., ((0,0,0), (0,1,0)) : O ((0,1,0), (0,0,0)) : X)
        """
        nodes, edges = [], []
        for ix in range(n * m):
            y, x = ix // m, ix % m
            nodes.append((x, y, 0))
            if x < m - 1:
                edges.append(((x, y, 0), (x + 1, y, 0)))
            if y < n - 1:
                edges.append(((x, y, 0), (x, y + 1, 0)))
        return nodes, edges

    def _init_graph(self, n, m, idx_netlists, g):
        if g:
            nodes, edges = g.nodes, g.edges
            constraints = g.graph
        else:
            self._get_2d_gridgraph(n, m)
            constraints = {}

        self.add_nodes_from(nodes, terminal=-1)
        self.add_edges_from(edges, weight=1)
        self.node_list, self.node_feats = list(zip(*self.nodes(data=True)))
        self.graph = constraints

        if g:
            self.netlists = self._extract_netlists(g)
            shape = g.dim
            if len(shape) == 2:
                n, m = shape
            elif len(shape) == 3:
                l, n, m = shape
            else:
                raise NotImplementedError
        else:
            shape = (n, m)
            self.netlists = {t + 1: [self.node_list[i] for i in net] for t, net in enumerate(idx_netlists)}

        for t, terminals in self.netlists.items():
            for term in terminals:
                self.nodes[term]["terminal"] = t

        self.edge_list = list(self.edges(data=True))
        self.n, self.m = n, m
        self.l = l if len(shape) == 3 else 1
        self.shape = shape

        self.terminals = self._get_terminals()
        self.n_nets = len(self.netlists)

    def _get_terminals(self):
        """
        find all terminal nodes without repetition.
        """
        return list(set(chain(*self.netlists.values())))

    def _extract_netlists(self, g):
        netlists = defaultdict(list)
        for node, attrs in g.nodes(data=True):
            if not attrs["is_terminal"]:
                continue
            netlists[attrs["tree"]].append(node)
        return netlists

    def update_edge_cost(self, c):
        """
        update edge cost by using node and edge weight.
        """
        attrs = {}
        for t1, t2 in self.edges:
            cost_from_edge, cost_from_node = self.edges[t1, t2]["weight"], (c[t1] + c[t2]) / 2
            attrs[(t1, t2)] = {"cost": (cost_from_edge + cost_from_node) / 2}
        nx.set_edge_attributes(self, attrs)


def transform_nx_graph(graph):
    """
    to be fixed.
    """
    g = nx.Graph()
    g.add_nodes_from(graph.nodes)

    for st_id in graph.edgelists.keys():
        neighbors = graph.edges[st_id]
        for nei in neighbors:
            if st_id > nei:
                continue  # remove duplication
            u, v = graph.nodes[st_id], graph.nodes[nei]
            g.add_edge(u, v)
    return g


def visualize_3d(G, STs, save=False, file_name="generated_graph_3d.png", congestion=False):

    selected_edges = list(chain(*[ed for ed, _ in STs]))
    terminals = np.array(G.terminals)
    terminals_each_level = [terminals[terminals[..., 2] == l] for l in range(G.l)]
    frame = np.array([e for e in G.edges()])
    labels = {n: "" if v["terminal"] == -1 else v["terminal"] for n, v in G.nodes(data=True)}
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]  # , ...

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    for ed in frame:
        ax.plot(*ed.T, "-", color="tab:gray", alpha=0.25)
    for ed in np.array(selected_edges):
        ax.plot(*ed.T, color="black", linewidth=2, alpha=0.9)
    for i, term in enumerate(terminals_each_level):
        ax.scatter(*term.T, s=120, color=colors[i])
    for k, v in labels.items():
        ax.text(*k, v, fontsize=12)

    ax.view_init(60, -60)  # pov (vertical, horizontal)

    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        ax.xaxis.set_ticks(list(range(G.n)))
        ax.yaxis.set_ticks(list(range(G.m)))
        ax.zaxis.set_ticks(list(range(G.l)))

        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("layer")

    _format_axes(ax)
    fig.tight_layout()

    if save:
        dir_name = "images"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.title("total edges : {e}".format(e=len(set(selected_edges))))
        try:
            plt.savefig(os.path.join(dir_name, file_name))
        except FileNotFoundError:
            file_dir = file_name.split("/")[0]
            os.makedirs(os.path.join(dir_name, file_dir), exist_ok=True)
            plt.savefig(os.path.join(dir_name, file_name))
        print("{f} is saved !".format(f=file_name))
    plt.show()
    plt.close()


def visualize(G, STs, save=False, file_name="generated_graph.png", congestion=False):
    x, y, z = G.n, G.m, G.l
    axes = plt.figure(constrained_layout=True, figsize=(z * 7, 7)).subplots(1, z, sharex=True, sharey=True)
    selected_edges = list(chain(*[ed for ed, _ in STs]))
    if not isinstance(axes, Iterable):
        axes = [axes]

    plt.title("total edges : {e}".format(e=len(set(selected_edges))))

    for k, ax in enumerate(axes):
        if G.__class__.__name__ == "GridGraph":
            pos = dict((n, n[:2]) for n in G.node_list)
            g = G.subgraph([n for n in G.node_list if n[-1] == k]) if len(G.shape) == 3 else G
        else:
            pos = nx.spring_layout(G)
            g = G
        g.terminals = [n for n in G.terminals if n[-1] == k] if len(G.shape) == 3 else G.terminals

        selected_edges_k = (
            [e for ed, _ in STs for e in ed if e[0][-1] == k and e[1][-1] == k] if len(G.shape) == 3 else selected_edges
        )
        node_size = 100 / math.sqrt(len(g))
        width = 30 / int(len(g) ** (1 / 4))

        # draw terminal
        labels = {n: "" if v["terminal"] == -1 else v["terminal"] for n, v in g.nodes(data=True)}
        nx.draw_networkx_nodes(
            g,
            pos=pos,
            ax=ax,
            nodelist=g.terminals,
            node_size=node_size * 10,
            node_color="g",
            node_shape="^",
        )
        nx.draw_networkx_labels(
            g,
            pos=pos,
            ax=ax,
            labels=labels,
        )
        # generated edges
        nx.draw_networkx_edges(
            g, pos=pos, ax=ax, edgelist=selected_edges_k, width=width, edge_color="red", style="solid"
        )

        # find congested nodes and visualize them
        if congestion:
            selected_nodes = Counter(list(chain(*[no for _, no in STs])))
            congests = [node for node, num in selected_nodes.items() if num > 1]
            # draw terminal
            nx.draw_networkx_nodes(
                g, pos=pos, ax=ax, nodelist=congests, node_size=node_size * 20, node_color="blue", node_shape="*"
            )
        ax.xaxis.set_ticks(list(range(G.n)))
        ax.yaxis.set_ticks(list(range(G.m)))
        ax.grid(True)
    if save:
        dir_name = "images"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.title("total edges : {e}".format(e=len(set(selected_edges))))
        try:
            plt.savefig(os.path.join(dir_name, file_name))
        except FileNotFoundError:
            file_dir = file_name.split("/")[0]
            os.makedirs(os.path.join(dir_name, file_dir), exist_ok=True)
            plt.savefig(os.path.join(dir_name, file_name))
        print("{f} is saved !".format(f=file_name))
    plt.show()
    plt.close()


def get_sample():
    idx_netlists = [[1, 4], [14, 10], [9, 25, 27]]  # second-order sample
    G = GridGraph(n=5, m=6, idx_netlists=idx_netlists)
    return G


def get_complete(given_g, selected_nodes):
    """
    construct complete graph of selected nodes of which edge weight is shortest path length.

    Parameters
    -------------
    given_g : nx.Graph(). given graph
    selected_nodes : list of nodes

    Returns
    ------------
    complete_g : nx.Graph(). It has two attributes which are 'shortest-path' and 'weight'.
                    'shortest-path' is a sequential list of nodes.
                    'weight' means shortest path length.
    """

    complete_g = nx.complete_graph(selected_nodes)
    attrs = {}
    for t1, t2 in combinations(selected_nodes, 2):
        attrs[(t1, t2)] = {
            "shortest-path": nx.shortest_path(given_g, source=t1, target=t2, weight="cost"),
            "weight": nx.shortest_path_length(given_g, source=t1, target=t2, weight="cost"),
        }
    nx.set_edge_attributes(complete_g, attrs)
    return complete_g


def find_mst(g, weight):
    return nx.minimum_spanning_tree(g, weight=weight)


def restore_graph(given_g, virtual_g):
    g = nx.Graph()

    def extract_edges(nodes):  # extract edge (node pair) from path (sequential node list).
        return [(nodes[ix - 1], nodes[ix]) for ix in range(1, len(nodes))]

    for node_path in nx.get_edge_attributes(virtual_g, "shortest-path").values():
        edges = extract_edges(node_path)
        subgraph = given_g.edge_subgraph(edges)
        g.add_nodes_from(subgraph.nodes(data=True))
        g.add_edges_from(subgraph.edges(data=True))
    return g


def remove_inessentials(cand_g, terminals):
    """
    remove redundant nodes and edges to ensure that all leaf nodes are terminal nodes.

    Parameters
    ------------
    cand_g : nx.Graph(). Steiner Tree candidate graph.
    terminals : list of tuples. It contains terminal nodes.
    """
    leafnodes = [n for n, dg in cand_g.degree() if dg == 1]
    not_terminals = leafnodes
    while not_terminals:
        node = not_terminals.pop()
        if node in terminals:
            continue
        # if leaf node is not terminal, this node will be removed.
        neighbors = list(cand_g.neighbors(node))
        assert len(neighbors) == 1  # degree of leaf node is 1
        while neighbors:
            nei = neighbors.pop()
            cand_g.remove_edge(node, nei)
            # again not terminal leaf node
            if (cand_g.degree[nei] == 1) and (nei not in terminals):
                not_terminals.append(nei)
        cand_g.remove_node(node)
    return cand_g
