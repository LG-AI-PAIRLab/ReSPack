import multiprocessing as mp
import os
import random
import time
from copy import deepcopy

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from stppgen.utils.graphio import (draw_graph_3d, draw_graph_3d_ver2, read_problem_from_file,
                                   read_solution_from_file, write_edges, 
                                   write_terminals, write_solution,
                                   write_meta, write_constraints,
                                   draw_graph_1layer)


verbose = False

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            if verbose:
                endl = '\n' if method.__name__ in ['build_tree', 'generate'] else ' '
                print(f'{method.__name__} {(te - ts):.2f} s', end=endl)
        return result
    return timed


class GridSTP:
    def __init__(self, dim: list, max_ntrees: int, max_nterminals: list,
                 **kwargs) -> None:
        self.dim = dim
        self.r_min = kwargs['r_min'] if 'r_min' in kwargs.keys() else 2
        self.r_max = kwargs['r_max'] if 'r_max' in kwargs.keys() else 5
        self.max_ntrees = max_ntrees
        self.max_nterminals = max_nterminals
        self.color_graph = kwargs['color_graph'] if 'color_graph' in kwargs.keys() else False
        self.random_seed = kwargs['random_seed'] if 'random_seed' in kwargs.keys() else None
        self.method = kwargs['method'] if 'method' in kwargs.keys() else None
        self.kwargs = kwargs
        self.g = self.init_graph(self.dim)

    def init_graph(self, dim):
        g = nx.grid_graph(dim)
        g.dim = dim

        # init node attributes
        for t in g.nodes:
            g.nodes[t]['is_terminal'] = False
            g.nodes[t]['tree'] = 0  # 0 is background

        # init edge attributes
        for e in g.edges:
            g.edges[e]['tree'] = 0  # 0 is background
            g.edges[e]['weight'] = 1  # 0 is background
        
        # init tree constraints
        g.graph['constraints'] = {}
        
        return g

    def generate(self):
        idx_tree = 1
        for i in range(self.max_ntrees):
            g_path = self.g
            try:
                is_success = self.build_tree(g_path, self.max_nterminals[i], idx_tree)
            except ValueError:
                break
            if is_success:
                idx_tree += 1

    def build_tree(self, g_path, n, idx_tree):
        is_success = self._build_tree(g_path, n, idx_tree)
        return is_success

    def _build_tree(self, g_path, n, idx_tree) -> bool:
        return NotImplementedError()

    def not_allow_terminal_via(self, g, x):
        if len(self.dim) == 3:
            for j in range(1, self.dim[0]):
                src = (x[0], x[1], j-1)
                trg = (x[0], x[1], j)
                try:
                    g.edges[(src, trg)]['weight'] = np.inf
                except Exception as e:
                    pass
        return g

    def get_terminal_candidates(self, g_path, freenodes, terminals, idx_tree, n=None):
        tree_nodes = deepcopy(terminals)
        if len(terminals) > 0:
            tree_nodes += [k for k, v in nx.get_node_attributes(g_path, 'tree').items() if v == idx_tree]
            freenodes = self.filter_grid_by_radius(freenodes, tree_nodes, terminals[0])
        else:
            freenodes = self.filter_grid_by_radius(freenodes, tree_nodes)
        n = len(freenodes) if n is None else min(len(freenodes), n)
        if self.random_seed is not None:
            random.seed(self.random_seed)
        freenodes = random.sample(freenodes, n)
        return freenodes

    def cutoff_tree_on_graph(self, g, idx_tree):
        nodes = [n for n in g.nodes if g.nodes[n]['tree'] == idx_tree]
        for e in g.edges(nodes):
            g.edges[e]['weight'] = np.inf
    
    def filter_grid_by_radius(self, candidates, center_pts_in=[], center_pts_out=None):
        if len(center_pts_in) == 0 and center_pts_out is not None:
            return candidates
        else:
            candidates_arr = np.array(candidates)
            idx = np.repeat(True, len(candidates))
            if len(center_pts_in) > 0:
                center_pts_in = np.array(center_pts_in)
                dist_in, _ = KDTree(center_pts_in).query(candidates_arr, k=1, p=1, distance_upper_bound=self.r_min + 0.01)
                idx_in = dist_in > self.r_min
                idx = idx & idx_in
            
            if center_pts_out is not None:
                center_pts_out = np.array([center_pts_out])
                dist_out, _ = KDTree(center_pts_out).query(candidates_arr, k=1, p=1, distance_upper_bound=self.r_max + 0.01)
                idx_out = dist_out < self.r_max
                idx = idx & idx_out
            return [tuple(x) for x in candidates_arr[idx].tolist()]

    def draw_graph(self, x=3, y=3, save="sample.png"):
        if len(self.dim) > 3:
            raise NotImplementedError('it cannot display the graph.')
        z = self.dim[0]
        draw_graph_3d(self.g, x, y, z, save, color=False)
        draw_graph_3d_ver2(self.g, x, y, z, save, color=False)

    @property
    def n_tree(self):
        return max([self.g.nodes[x]['tree'] for x in self.g.nodes])
            
    def save_problem_txt(self, path):
        g, dim, n_tree = self.g, self.dim, self.n_tree
        attr_tree = nx.get_node_attributes(g, 'tree')
        attr_terminal = nx.get_node_attributes(g, 'is_terminal')
        write_meta(g, path, dim, n_tree, attr_tree, attr_terminal)
        write_edges(g, path)
        write_terminals(g, path, n_tree, attr_tree, attr_terminal)
        write_constraints(g, path, n_tree, attr_tree)
        print(f"[EOF]", file=open(path, 'a'))

    def save_solution_txt(self, path):
        g, n_tree = self.g, self.n_tree
        attr_tree = nx.get_edge_attributes(g, 'tree')
        write_solution(path, n_tree, attr_tree)
        print(f"[EOF]", file=open(path, 'a'))

    def read_STP_from_files(self, path_prob, path_sol=None):
        g = read_problem_from_file(path_prob)
        if path_sol is not None:
            g = read_solution_from_file(g, path_sol)
        self.g = g


class BaseDataGenerator:
    def __init__(self, n, method, dim,
                 ntrees,
                 nterminals,
                 r_min=2, r_max=10,
                 save_to_dir=None, save_plot=False,
                 overwrite=False, **kwargs):
        self.n = n
        self.method = method
        self.dim = dim
        self.ntrees = ntrees
        self.nterminals = nterminals
        self.r_min = r_min
        self.r_max = r_max
        self.save_to_dir = save_to_dir
        self.random_seed = kwargs['random_seed'] if 'random_seed' in kwargs.keys() else None
        self.save_plot = save_plot
        self.overwrite = overwrite
        self.kwargs = kwargs
    
    def init_stp_generator(self, dim, max_ntrees, max_nterminals, idx=None):
        return NotImplementedError

    def generate_instance(self, idx):
        dim = self.dim
        if self.random_seed is not None:
            random_seed = self.random_seed + idx
        else:
            random_seed = None
        self.kwargs['random_seed'] = random_seed
        max_ntrees = self.ntrees.sample(random_state=random_seed)[0]
        max_nterminals = self.nterminals.sample(n=max_ntrees, random_state=random_seed)
        stp = self.init_stp_generator(dim, max_ntrees, max_nterminals, idx)

        path_for_problem = os.path.join(self.save_to_dir, f'problems/p_{idx}.txt')
        path_for_solution = os.path.join(self.save_to_dir, f'solutions/s_{idx}.txt')

        if not os.path.exists(path_for_solution) or self.overwrite:
            stp.generate()
            stp.save_problem_txt(path_for_problem)
            stp.save_solution_txt(path_for_solution)
            if self.save_plot:
                path_for_figure = os.path.join(self.save_to_dir, f'solutions/f_{idx}.png')
                stp.draw_graph(10, 10, save=path_for_figure)

    def run(self, ncores=4, start=0, end=None):
        end = self.n if end is None else end
        if self.save_to_dir is not None:
            if not os.path.exists(self.save_to_dir):
                os.makedirs(self.save_to_dir)
            os.makedirs(os.path.join(self.save_to_dir, 'problems'), exist_ok=self.overwrite)
            os.makedirs(os.path.join(self.save_to_dir, 'solutions'), exist_ok=self.overwrite)

        if ncores > 1:
            pool = mp.Pool(ncores)
            jobs = []
            for i in range(start, end, 1):
                jobs.append(pool.apply_async(self.generate_instance, (i, )))
            for j in jobs:
                j.get()
            pool.close()
            pool.join()
        else:
            for i in range(start, end, 1):
                self.generate_instance(i)
                