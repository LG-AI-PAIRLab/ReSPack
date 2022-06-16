import numpy as np
import networkx as nx

from networkx import shortest_path_length as spl
from itertools import combinations
from scipy.spatial import KDTree


class Obstacle:
    def __init__(self, dim, shape):
        self.dim = dim
        self.shape = shape
        
    def generate(self, random_seed=None):
        while True:
            o = self._generate_objects(self.dim, self.shape, random_seed)
            if o != False:
                break
        return o
    
    def _generate_objects(self, dim, shape, random_seed=None):
        """dim (z, y, x)"""
        pos = np.array([np.random.randint(n) for n in list(dim)[::-1]])
        object_nodes = []
        object_bound = []
        canvas_limit = [dim[2] - 1, dim[1] - 1]
        for i in range(shape[0] + 2):
            for j in range(shape[1] + 2):
                x = pos[0:2] + np.array([i, j])
                x = tuple(x) + (pos[2], )
                if i > 0 and j > 0 and i < shape[0] + 1 and j < shape[1] + 1:
                    object_nodes.append(x)
                object_bound.append(x)
                # check canvas out
                if sum([a > b for a, b in zip(x[:2], canvas_limit)]) > 0:
                    return False
        
        obj = {
            'center': pos + np.array([1, 1, 0]),
            'points': object_nodes,
            'boundary': set(object_bound)
        }
        
        return obj


class ObstacleAssigner:  
    def __init__(self, dim, scale, random_seed=None):
        self.dim = dim
        self.scale = scale
        self.random_seed = random_seed
        self.random = np.random.RandomState(self.random_seed)
    
    def assign_obstacles(self):
        obstacle_large = Obstacle(dim=self.dim, shape=(4, 4) * self.scale)
        obstacle_small = Obstacle(dim=self.dim, shape=(1, 2) * 1)
        
        iter = 0
        max_iter = 1000
        num_obj = 0
        obs_ratio = 0
        obj_list = []
        while obs_ratio < 0.5:
            p = self.random.uniform()
            if p < 0.5:
                o = obstacle_large.generate(random_seed=self.random_seed + iter)
            else:
                o = obstacle_small.generate(random_seed=self.random_seed + iter)
            
            if o == False:
                continue
            
            if len(obj_list) > 0:
                n_intersect = 0
                for prev in obj_list:
                    intersect = set(o['boundary']).intersection(prev['boundary'])
                    n_intersect += len(intersect)
                if n_intersect == 0:
                    obj_list.append(o)
                    num_obj += 1
                    obs_ratio = sum([len(x['points']) for x in obj_list]) / np.prod(self.dim)
            else:
                obj_list.append(o)
                num_obj += 1
                obs_ratio = sum([len(x['points']) for x in obj_list]) / np.prod(self.dim)
            iter += 1
            if iter > max_iter:
                break
        return obj_list
        

def init_obstacles(dim, g, scale, random_seed=None):
    # obstacles
    obs_assigner = ObstacleAssigner(dim, scale, random_seed)
    obs_list = obs_assigner.assign_obstacles()
    for obs in obs_list:
        for n in obs['points']:
            g.remove_node(n)
    return g


class TreeMarginConstraints:
    def __init__(self, **kwargs):
        self.max_margin_size = kwargs['margin_max']
        self.random_seed = kwargs['random_seed']
        self.margins = self.margin_size()
        
    def margin_size(self):
        random = np.random.RandomState(self.random_seed)
        margins = random.randint(low=1, high=self.max_margin_size + 1, size=10 ** 4)
        return margins.tolist()
    
    @staticmethod
    def _find_nodes_nearby(tree: list, nodes: list, margin: int):
        nodes = np.array(nodes)
        dist, _ = KDTree(np.array(tree)).query(nodes, k=1, p=1, distance_upper_bound=margin + 0.01)
        idx = dist <= margin
        return [tuple(x) for x in nodes[idx].tolist()]
    
    def assign_margin_constraint(self, g, idx_tree):
        nodes = g.nodes()
        tree = [n for n in g.nodes if g.nodes[n]['tree'] == idx_tree]
        margin = self.margins.pop()
        if margin > 0:
            margin_nodes = self._find_nodes_nearby(tree, nodes, margin=margin)
            nodes_used = [n for n in set(margin_nodes) - set(tree) if g.nodes[n]['tree'] != 0]
            if len(nodes_used) > 0:
                return False
            # update edges
            for e in g.edges(margin_nodes):
                g.edges[e]['weight'] = np.inf
                g.edges[(e[1], e[0])]['weight'] = np.inf
            for n in margin_nodes:
                g.nodes[n]['margin'] = True
        # update graph
        self.update_tree_contraints(g, idx_tree, margin)
        return True
            
    def update_tree_contraints(self, g, idx_tree, margin):
        try:
            if g.graph['constraints'][idx_tree]:
                g.graph['constraints'][idx_tree]['margin'] = margin
            else:
                g.graph['constraints'][idx_tree] = {'margin': margin}    
        except (KeyError, AttributeError):
            g.graph['constraints'][idx_tree] = {'margin': margin}
        

class TreeRadiusConstraints:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def assign_radius_constraint(self, g, idx_tree):
        tree_nodes = [n for n in g.nodes if g.nodes[n]['tree'] == idx_tree]
        terminals = [n for n in tree_nodes if g.nodes[n]['is_terminal']]
        tree = g.subgraph(tree_nodes)
    
        paths = list(combinations(terminals, 2))
        lengths = {(u, v): self.spl(tree, source=u, target=v) for u, v in paths}
        lengths = {k: v for k, v in lengths.items() if v < np.inf}
        
        ordered_paths = sorted(list(lengths.keys()), key=lambda p: -lengths[p])
        max_length = lengths[ordered_paths[0]]

        self.update_tree_constraints(g, idx_tree, max_length)
        return True
    
    def spl(self, tree, source, target):
        try:
            return nx.shortest_path_length(tree, source=source, target=target)
        except nx.exception.NetworkXNoPath:
            return np.inf

    def update_tree_constraints(self, g, idx_tree, radius):
        try:
            if g.graph['constraints'][idx_tree]:
                g.graph['constraints'][idx_tree]['radius'] = radius
            else:
                g.graph['constraints'][idx_tree] = {'radius': radius}
        except (KeyError, AttributeError):
            g.graph['constraints'][idx_tree] = {'radius': radius}
