import random
from itertools import combinations
from heuristic.utils import get_bbox
from heuristic.routing import Lee, TwoApprox
from common.algorithm import Algorithm


class Heuristics(Algorithm):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        print(
            "route : {r} / ripup and reroute : {rrr}".format(
                r=self.route, rrr=self.rrr
            )
        )

    def init_graph(self, G):
        """
        initialize graph.
        Parameters
        -------------
        G : class. e.g., GridGraph()
        """
        super().init_graph(G)
        self.max = len(G) ** 2  ###### MAX for BFS
        self._init_edge_cost()
        self._init_hyperparam()
        
    def init_seed(self, sd=None):
        import time
        if sd is None:
            self.seed = time.time()
        else:
            self.seed=sd
        random.seed(self.seed)

    def _init_hyperparam(self):
        p_fac, h_fac= 0.8, 0.1
        self.p_fac, self.h_fac = p_fac, h_fac

    def _init_edge_cost(self, base=1):
        self.b = {n: base for n in self.G.nodes}  # a single layer.
        self.p = {n: base for n in self.G.nodes}
        self.h = {n: base for n in self.G.nodes}
        self.c = {n: (self.b[n] + self.h[n]) * self.p[n] for n in self.G.nodes}  # c = (b+h) * p
        self.occ = {n: 0 for n in self.G.nodes}
        self.cap = {n: 1 for n in self.G.nodes}  # available numbers of nets
        self.mar = {n: 0 for n in self.G.nodes}

    def _init_margin(self,):
        self.mar = {n: 0 for n in self.G.nodes}

    def net_order(self):
        """
        make an order of nets.

        Returns
        -----------
        ordered list of net ids
        """
        if self.verbose:
            print("net ordering..... (random order)")
        order = list(self.G.netlists.keys())
        random.shuffle(order)
        return order
        
    def STP_routing(self, net_id):
        """
        routing algorithm for a single Steiner Tree
        """
        if self.verbose:
            print("Steiner Tree routing..... ({})".format(net_id))
        if self.route == "lee":
            st_edges, st_nodes = Lee(self.G, net_id, self.c)
        elif self.route == "2approx":
            st_edges, st_nodes = TwoApprox(self.G, net_id, self.c)
        else:
            print("{r} is not appropriate routing algorithm".format(r=self.route))
        return st_edges, st_nodes

    def check_spacing(self, prev_margins, new_nodes, n_space):
        """
        check spacing constraints and return flag and margin node set.
        Parameters
        -------------
        prev_margins    : dict of set.  
        new_nodes       : list of nodes
        n_space         : scalar.

        Returns
        -------------
        is_violated     : boolean.
        info            : set of nodes. it means nodes <= n_space
        """
        from functools import reduce
        violated = []
        # check if new path violates previous margin nodes
        prev_all_margins = list(reduce(lambda x, y: x|y, prev_margins.values(), set()))
        for nn in new_nodes:
            if nn in prev_all_margins:
                violated.append(nn)

        def get_margins(nodes): # find a set of neighborhoods
            margins = nodes.copy()
            for n in nodes:
                for nei in self.G.neighbors(n):
                    margins.add(nei)
            return margins

        new_margins = set(new_nodes)    # starts from original route
        for _ in range(n_space):    # find n-hop neighborhoods
            new_margins = get_margins(new_margins)  
        new_margins -= set(new_nodes)   # remove route and remain only margins

        is_violated, info = False if not violated else True, new_margins
        return is_violated, info

    def check_radius(self, new_st, terminals, radius):
        is_violated, max_radius = False, -1
        
        new_edges, _ = new_st

        import networkx as nx
        nx.set_edge_attributes(self.G, float('inf'), 'routed')  # initialize all edges by inf

        for ed in new_edges:    # selected edges = 1
            self.G.edges[ed]['routed'] = 1 

        # find max length (radius of steiner tree)
        for t1, t2 in combinations(terminals, 2):
            r = nx.shortest_path_length(self.G, t1, t2, 'routed')   # find shortest path length
            if max_radius < r:
                max_radius = r

        # check radius constraints (max_radius <= radius)
        is_violated, info = radius < max_radius, max_radius
        return is_violated, info
    
    def is_violate_constraints(self, prev_margins, new_st, net_id):
        _, st_nodes = new_st
        # no constraints
        if ('constraints' not in self.G.graph.keys()) or (not self.G.graph['constraints']): 
            return False, (-1, set())   # default
        const = self.G.graph['constraints'][net_id]
        result_r, info_r = self.check_radius(new_st, self.G.netlists[net_id], const['radius'])
        result_s, info_s = self.check_spacing(prev_margins, st_nodes, const['margin'])

        is_violated =  result_r or result_s
        return is_violated, (info_r, info_s)


    def naive(self, order):
        new_STs, congestion = {}, False
        prev_margins = {}
        for net_id in order:
            st_edges, st_nodes = self.STP_routing(net_id)
            new_STs[net_id] = (st_edges, st_nodes)
            is_violated, (_, new_margins) = self.is_violate_constraints(prev_margins, (st_edges, st_nodes), net_id)
            is_shared = self.update_p(st_nodes, 1, "route") # only check congestion
            prev_margins[net_id] = new_margins
            if is_shared or (not st_nodes) or is_violated:
                congestion = True
                break
            self.update_c(st_nodes, unblock=False)
            self.update_c(new_margins, unblock=False)   # block margin
            
        return new_STs, congestion

    def update_c(self, st_nodes, unblock=True):
        for n in st_nodes:
            if unblock:
                self.c[n] = (self.b[n] + self.h[n]) * self.p[n]
            else:
                self.c[n] = float('inf')

    def update_p(self, st_nodes, trial, type="route", is_margin=False):
        assert type in ["route", "ripup"]
        congestion = False
        dev = 1 if type == "route" else -1

        for n in st_nodes:
            if is_margin:   # margin
                self.mar[n] += dev
            else:           # route
                self.occ[n] += dev
            # 1) congestion (path - path)
            route_intersect = self.occ[n] > self.cap[n]
            # 2) prev route intersects margin or path intersects prev margin (path - margin)
            margin_pass_route = bool(self.occ[n] and self.mar[n])
            if type == "route" and (not congestion) and (route_intersect or margin_pass_route):
                congestion = True
            self.p[n] += dev * trial * self.p_fac
        return congestion

    def update_h(self, st_nodes, trial):
        for n in st_nodes:
            self.h[n] += self.h_fac
    
    def STPP_routing(self, order):
        """
        route all Steiner Trees without considering other trees (i.e., not update cost c)

        Returns
        ----------
        new_STs : dictionary of tuples which are (st_edges, st_nodes).
        congestion : Boolean. It means occurence of congestion.
        """
        new_STs, traces, congestion = {}, set(), False
        prev_margins = {}
        for net_id in order:
            st_edges, st_nodes = self.STP_routing(net_id)
            is_violated, (_, new_margins) = self.is_violate_constraints(prev_margins, (st_edges, st_nodes), net_id)
            new_STs[net_id] = (st_edges, st_nodes)
            prev_margins[net_id] = new_margins
            is_shared = self.update_p(st_nodes, 1, "route") # route (in STPP, trial=1)
            is_shared2 = self.update_p(new_margins, 1, "route", is_margin=True) # margin (in STPP, trial=1)
            if (not congestion) and (is_shared or is_shared2 or is_violated):
                congestion = True
        return new_STs, congestion, prev_margins

    def ripup_and_rerouting(self, new_STs, order, congestion, new_MGs):
        trial = 1
        while congestion and not self.exit_cond(trial):
            if self.verbose:
                print("iter {} : rip-up and retry".format(trial))
            congestion = False
            prev_STs, new_STs = new_STs, {}
            prev_MGs, new_MGs = new_MGs, {}
            for net_id in order:
                prev_st, prev_mg = prev_STs[net_id], prev_MGs[net_id]
                _, prev_nodes = prev_st
                # rip-up previous route & margin
                self.update_p(prev_nodes, trial, "ripup")  # route
                self.update_p(prev_mg, trial, "ripup", is_margin=True)  # margin
                self.update_c(list(set(prev_nodes) | set(prev_mg)))
                new_st = self.STP_routing(net_id)
                new_STs[net_id] = new_st
                is_violated, (_, new_mg) = self.is_violate_constraints(new_MGs, new_st, net_id)
                _, new_nodes = new_st
                is_shared = self.update_p(new_nodes, trial + 1, "route")   # route
                is_shared2 = self.update_p(new_mg, trial + 1, "route", is_margin=True)  # margin
                new_MGs[net_id] = new_mg
                if (not congestion) and (is_shared or is_shared2 or is_violated):
                    congestion = True
                if self.rrr == "pathfinder":
                    self.update_h(new_nodes, trial + 1)    # route
                    self.update_h(new_mg, trial + 1)    # margin
                self.update_c(list(set(new_nodes) | set(new_mg)))
            trial += 1
        return (new_STs, trial) if (not congestion) and (not self.exit_cond(trial)) else (None, trial)

    def run(self):
        # net ordering
        order = self.net_order()

        # routing
        if self.rrr == "naive": # Sequential
            iterations = 1
            while not self.exit_cond(iterations):
                new_STs, congestion = self.naive(order)
                if not congestion:
                    break
                order = self.net_order()
                self._init_edge_cost()
                iterations += 1

            if self.exit_cond(iterations):  # exceed iter. limits
                new_STs = None

        else:   # PathFinder
            new_STs, congestion, margins = self.STPP_routing(order) # route all nets w/o considering congestion
            # ripup and rerouting
            new_STs, iterations = self.ripup_and_rerouting(new_STs, order, congestion, margins)

        result = {
            "solution": new_STs,
            "iteration": iterations,
        }
        return result
