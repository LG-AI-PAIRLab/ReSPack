class Algorithm:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def init_graph(self, graph):
        """
        initialize graph.
        Parameters
        -------------
        graph : class. e.g., GridGraph()
        """
        self.G = graph
        self.n_nodes = len(self.G)

    def init_seed(self, sd=None):
        import time

        if sd is None:
            self.seed = time.time()
        else:
            self.seed = sd

    def _init_edge_cost(self, base=1):
        self.c = {n: base for n in self.G.nodes}  # a single layer.

    def exit_cond(self, trial, type=None, max_iter=200):
        if type is None and trial < max_iter:
            return False
        return True

    def run(self):
        result = {
            "solution": None,
        }
        return result
