# TODO: move to generator.py & rename
import multiprocessing as mp
import numpy as np
import networkx as nx
from stppgen.stpsolution import *
from stppgen.utils.distribution import Poisson
from stppgen.utils.graphio import read_problem_from_file, read_solution_from_file
from stppgen.constraints import init_obstacles, TreeMarginConstraints, TreeRadiusConstraints


class SequentialGenerator(GridSTPfast):
    """
    It interatively generates STPP instances (problem & solution) from an 
    initial STPP instance with MCMC sampling idea
    """
    def __init__(self, *args, **kwargs):
        self.dim = kwargs['dim']
        self.ratio_rmv = 0.05
        self.random_nterminals = False
        self.kwargs = kwargs
        self.r_min = kwargs['r_min']
        self.r_max = kwargs['r_max']
        self.random_seed = kwargs['random_seed']
        self.init_graph()
        self.fail_cnt = 0
        
    def get_dimension(self):
        nodes = np.array(self.g.nodes)
        x, y, z = np.max(nodes, axis=0) + 1
        return [z, x, y]
        
    def init_graph(self):
        g = read_problem_from_file(self.kwargs['path_problem'])
        self.g = read_solution_from_file(g, self.kwargs['path_solution'])
        
    def generate(self, nsteps=None):
        self.step()
        
    def step(self):
        tree_rmv_idx = self.choice_trees_to_remove()
        tree_rmv = self.remove_tree_by_index(tree_rmv_idx)
        self.rebuild_trees(tree_rmv)
    
    def choice_trees_to_remove(self):
        ntree = max([self.g.nodes[x]['tree'] for x in self.g.nodes])
        n_rmv = max(int(ntree * self.ratio_rmv), 1)
        tree_rmv_idx = np.random.choice(np.arange(1, ntree + 1), n_rmv, replace=False)
        return tree_rmv_idx
    
    def remove_tree_by_index(self, tree_rmv_idx):
        attr_node = nx.get_node_attributes(self.g, 'tree')
        attr_edge = nx.get_edge_attributes(self.g, 'tree')
        # remove trees
        nterminals = []
        for idx in tree_rmv_idx:
            treenodes = [k for k, v in attr_node.items() if v == idx]
            treeedges = [k for k, v in attr_edge.items() if v == idx]
            # keep # terminals
            nterminals.append(len([n for n in treenodes if self.g.nodes[n]['is_terminal'] == True]))
            # remove nodes
            for n in treenodes:
                self.g.nodes[n]['tree'] = 0
                self.g.nodes[n]['is_terminal'] = False
            # remove edges
            for e in treeedges:
                self.g.edges[e]['tree'] = 0
                self.g.edges[e]['weight'] = 1
            # remove edges to tree
            for e in self.g.edges(treenodes):
                s, t = e
                if self.g.nodes[s]['tree'] == 0 and self.g.nodes[t]['tree'] == 0:
                    self.g.edges[e]['weight'] = 1
        return dict(zip(tree_rmv_idx, nterminals))
                
    def rebuild_trees(self, tree_rmv):
        for idx, nt in tree_rmv.items():
            if self.random_nterminals:
                nt = Poisson(self.kwargs['poilam']).sample(1)[0]
            is_success = self.make_a_tree(idx, nt)
    
    def make_a_tree(self, idx, nt):
        try:
            is_success = self.build_tree(self.g, nt, idx)
        except ValueError:
            return False
        else:
            return is_success
        
        
class SequentialGridSTPfastObstacle(SequentialGenerator):
    def __init__(self, *args, **kwargs):
        self.init_from_file=True
        super().__init__(*args, **kwargs)

class ConstrainedSequentialGridSTPfast(SequentialGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setmargin = TreeMarginConstraints(**kwargs)
        self.setradius = TreeRadiusConstraints(**kwargs)
    
    def init_graph(self):
        super().init_graph()
        for t in self.g.nodes:
            self.g.nodes[t]['margin'] = False
    
    def get_terminal_candidates(self, g_path, freenodes, terminals, idx_tree, n=None):
        freenodes = [n for n in freenodes if not self.g.nodes[n]['margin']]
        return super().get_terminal_candidates(g_path, freenodes, terminals, idx_tree, n)
    
    def _build_tree(self, g_path, n, idx_tree):
        is_success = super()._build_tree(g_path, n, idx_tree)
        is_ok = self.setmargin.assign_margin_constraint(g_path, idx_tree)
        if not (is_success and is_ok):  # add constraints
            self.setmargin.update_tree_contraints(g_path, idx_tree, margin=0)
        if is_success:
            self.setradius.assign_radius_constraint(g_path, idx_tree)
        return is_success
    
    def find_largest_component(self, g):
        nodes = [n for n in g.nodes if g.nodes[n]['tree'] == 0 and not g.nodes[n]['margin']]
        g_ = g.subgraph(nodes)
        largest_cc = max(nx.connected_components(g_), key=len)
        return list(largest_cc)