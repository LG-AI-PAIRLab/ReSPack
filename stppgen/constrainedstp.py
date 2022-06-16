from stppgen.constraints import init_obstacles, TreeMarginConstraints, TreeRadiusConstraints
from stppgen.stpsolution import *
from stppgen.utils.graphio import read_problem_from_file


class GridSTP2optObstacle(GridSTP2opt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def init_graph(self, dim):
        g = super().init_graph(dim)
        return init_obstacles(dim, g, scale=self.kwargs['obstacle'], random_seed=self.random_seed)


class GridSTPfastObstacle(GridSTPfast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def init_graph(self, dim):
        g = super().init_graph(dim)
        return init_obstacles(dim, g, scale=self.kwargs['obstacle'], random_seed=self.random_seed)


class ConstrainedGridSTPfast(GridSTPfast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setmargin = TreeMarginConstraints(kwargs['margin_max'])
    
    def _build_tree(self, g_path, n, idx_tree):
        is_success = super()._build_tree(g_path, n, idx_tree)
        if is_success:  # add constraints
            self.setmargin.assign_margin_constraint(g_path, idx_tree)
        return is_success


class ConstrainedGridSTP2opt(GridSTP2opt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setmargin = TreeMarginConstraints(kwargs['margin_max'])
    
    def _build_tree(self, g_path, n, idx_tree):
        is_success = super()._build_tree(g_path, n, idx_tree)
        if is_success:  # add constraints
            self.setmargin.assign_margin_constraint(g_path, idx_tree)
        return is_success


# for test radius constraints
class GridSTPfastRadius(GridSTPfast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        search_cutoff = self.r_max
        self.setradius = TreeRadiusConstraints(search_cutoff)

    def _build_tree(self, g_path, n, idx_tree):
        is_success = super()._build_tree(g_path, n, idx_tree)
        if is_success:
            radius_found = self.setradius.assign_radius_constraint(g_path, idx_tree)
        return (is_success and radius_found)


# for test radius constraints
class GridSTP2optRadius(GridSTP2opt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        search_cutoff = self.r_max
        self.setradius = TreeRadiusConstraints(search_cutoff)

    def _build_tree(self, g_path, n, idx_tree):
        is_success = super()._build_tree(g_path, n, idx_tree)
        if is_success:
            radius_found = self.setradius.assign_radius_constraint(g_path, idx_tree)
        return (is_success and radius_found)


class GridSTPfastObstacleandCostraint(GridSTPfast):
    def __init__(self, *args, **kwargs):
        self.init_base_dir = kwargs['init_base_dir']
        self.idx = kwargs['idx']
        super().__init__(*args, **kwargs)
        self.setmargin = TreeMarginConstraints(kwargs['margin_max'])
        self.setradius = TreeRadiusConstraints()
    
    def init_graph(self, dim):
        if self.init_base_dir:
            return self._init_graph_from_file_idx(dim, self.idx)
        else:
            g = super().init_graph(dim)
            for t in g.nodes:
                g.nodes[t]['margin'] = False
                g.nodes[t]['tree'] = 0
                g.nodes[t]['is_terminal'] = False
            return init_obstacles(dim, g, scale=self.kwargs['obstacle'], random_seed=self.random_seed)
    
    def _init_graph_from_file_idx(self, dim, idx):
        idx = self.idx
        path_problem = f"{self.init_base_dir}/problems/p_{idx}.txt"
        g = read_problem_from_file(path_problem)
        for t in g.nodes:
            g.nodes[t]['margin'] = False
            g.nodes[t]['tree'] = 0
            g.nodes[t]['is_terminal'] = False
        return g
    
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
    

class GridSTP2optObstacleandCostraint(GridSTP2opt):
    def __init__(self, *args, **kwargs):
        self.init_base_dir = kwargs['init_base_dir']
        self.idx = kwargs['idx']
        super().__init__(*args, **kwargs)
        self.setmargin = TreeMarginConstraints(**kwargs)
        self.setradius = TreeRadiusConstraints(**kwargs)
    
    def init_graph(self, dim):
        if self.init_base_dir:
            return self._init_graph_from_file_idx(dim, self.idx)
        else:
            g = super().init_graph(dim)
            for t in g.nodes:
                g.nodes[t]['margin'] = False
                g.nodes[t]['tree'] = 0
                g.nodes[t]['is_terminal'] = False
            return init_obstacles(dim, g, scale=self.kwargs['obstacle'], random_seed=self.random_seed)
    
    def _init_graph_from_file_idx(self, dim, idx):
        idx = self.idx
        path_problem = f"{self.init_base_dir}/problems/p_{idx}.txt"
        g = read_problem_from_file(path_problem)
        for t in g.nodes:
            g.nodes[t]['margin'] = False
            g.nodes[t]['tree'] = 0
            g.nodes[t]['is_terminal'] = False
        return g
    
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


class GridSTPfastObstacleandCostraint(GridSTPfast):
    def __init__(self, *args, **kwargs):
        self.init_base_dir = kwargs['init_base_dir']
        self.idx = kwargs['idx']
        super().__init__(*args, **kwargs)
        self.setmargin = TreeMarginConstraints(**kwargs)
        self.setradius = TreeRadiusConstraints()
    
    def init_graph(self, dim):
        if self.init_base_dir:
            return self._init_graph_from_file_idx(dim, self.idx)
        else:
            g = super().init_graph(dim)
            for t in g.nodes:
                g.nodes[t]['margin'] = False
                g.nodes[t]['tree'] = 0
                g.nodes[t]['is_terminal'] = False
            return init_obstacles(dim, g, scale=self.kwargs['obstacle'], random_seed=self.random_seed)
    
    def _init_graph_from_file_idx(self, dim, idx):
        idx = self.idx
        path_problem = f"{self.init_base_dir}/problems/p_{idx}.txt"
        g = read_problem_from_file(path_problem)
        for t in g.nodes:
            g.nodes[t]['margin'] = False
            g.nodes[t]['tree'] = 0
            g.nodes[t]['is_terminal'] = False
        return g
    
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
    
