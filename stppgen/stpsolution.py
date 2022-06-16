import networkx as nx
import numpy as np
from stppgen.utils.st import steiner_tree
from stppgen.base import GridSTP


class GridSTP2opt(GridSTP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_tree(self, g_path, n, idx_tree):
        terminals = self.sample_terminal_pts_on_comp(g_path, n, idx_tree)
        if len(terminals) < 2:
            return False
        try:
            st = steiner_tree(g_path, terminal_nodes=terminals, weight='weight')
        except Exception as e:
            return False
        else:
            self.update_tree_on_graph(g_path, st, terminals, idx_tree)
            self.cutoff_tree_on_graph(g_path, idx_tree)
            return True

    def sample_terminal_pts_on_comp(self, g_path, nterminals, idx_tree) -> list:
        terminals = []
        candidates = self.find_largest_component(g_path)
        candidates = [n for n in candidates if g_path.nodes[n]['tree'] == 0]
        for i in range(nterminals):
            cand_ = self.get_terminal_candidates(g_path, candidates, terminals, idx_tree)
            terminals += cand_[:1]
        return terminals

    def find_largest_component(self, g):
        nodes = [n for n in g.nodes if g.nodes[n]['tree'] == 0]
        g_ = g.subgraph(nodes)
        largest_cc = max(nx.connected_components(g_), key=len)
        return list(largest_cc)

    def update_tree_on_graph(self, g, subgraph, terminals, idx_tree):
        # update node attributes
        for t in subgraph.nodes:
            g.nodes[t]['tree'] = idx_tree

        for e in g.edges(subgraph.nodes):
            g.edges[e]['weight'] = np.inf
            g.edges[(e[1], e[0])]['weight'] = np.inf

        for t in terminals:
            g.nodes[t]['is_terminal'] = True
            self.not_allow_terminal_via(g, t)

        # update edge attributes
        for e in subgraph.edges:
            g.edges[e]['tree'] = idx_tree  # 0 is background


class GridSTPfast(GridSTP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_cnt = 0

    def _build_tree(self, g_path, n, idx_tree):
        if self.fail_cnt > 3:
            return False
        terminals = []
        for i in range(n):
            is_success, g_path, terminals = self.expand_tree(g_path, terminals, idx_tree)
            if is_success and i == 0:
                # update g_path for initial terminal node if tree is expanded
                g_path.nodes[terminals[0]]['tree'] = idx_tree
                g_path.nodes[terminals[0]]['is_terminal'] = True
                
        if len(terminals) > 1:
            self.cutoff_tree_on_graph(g_path, idx_tree)
            self.fail_cnt = 0
            return True
        else:
            self.fail_cnt += 1
            return False
        
    def expand_tree(self, g_path, terminals, idx_tree):
        MAX_TRY = 100
        freenodes = [n for n in g_path.nodes if g_path.nodes[n]['tree'] == 0]
        is_success = False
        if len(terminals) == 0:
            nodes_src = self.get_terminal_candidates(g_path, freenodes, terminals, idx_tree, n=MAX_TRY)
            for n in nodes_src:
                terminals_tmp = [n]
                nodes_trg = self.get_terminal_candidates(g_path, freenodes, terminals_tmp, idx_tree, n=10)
                for t in nodes_trg:
                    is_success, g_path = self.expand_tree_to_target(g_path, terminals_tmp, t, idx_tree)
                    if is_success:
                        terminals += [n, t]
                        break
                if is_success:
                    break
            if not is_success:
                raise ValueError
        else:
            nodes_trg = self.get_terminal_candidates(g_path, freenodes, terminals, idx_tree, n=10)
            for t in nodes_trg:  # max try 100
                is_success, g_path = self.expand_tree_to_target(g_path, terminals, t, idx_tree)
                if is_success:
                    terminals.append(t)
                    break
        return is_success, g_path, terminals

    def expand_tree_to_target(self, g_path, terminals, t, idx_tree):
        # choice junction node within the existing tree
        nodes = [k for k, v in nx.get_node_attributes(g_path, 'tree').items() if v == idx_tree]
        if len(nodes) == 0:
            nodes = [terminals[0]]
        is_connected = False
        path = self.find_path(g_path, set(nodes), t)
        # update graph when path exists
        if len(path) > 1:
            g_path = self.update_path_on_graph(g_path, path, idx_tree)
            is_connected = True
        return is_connected, g_path

    def find_path(self, g_path, u, v):
        try:
            length, path = nx.algorithms.shortest_paths.multi_source_dijkstra(g_path, u, v, weight='weight')
        except TypeError:
            pass
        except nx.exception.NetworkXNoPath:
            # no path
            path = []
        else:
            if length == np.inf:
                path = []
        return path

    def update_path_on_graph(self, g, path, idx_tree):
        g.nodes[path[0]]['tree'] = idx_tree
        for i, n in enumerate(path[1:]):
            e = (path[i], n)
            g.edges[e]['weight'] = np.inf
            g.edges[(n, path[i])]['weight'] = np.inf
            g.edges[e]['tree'] = idx_tree

            # update node attributes
            g.nodes[n]['tree'] = idx_tree
        g.nodes[n]['is_terminal'] = True
        g = self.not_allow_terminal_via(g, n)
        return g

