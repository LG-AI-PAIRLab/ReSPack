import os
import random
from stppgen.utils.graphio import read_problem_from_file, read_solution_from_file
from common.graph import GridGraph

class Loader:
    def __init__(self, shuffle=True, root_dir=None, sample_dir=None, n_samples=1000, n_split=1, div=0):
        self.sample_dir = sample_dir
        if not root_dir:
            root_dir = os.path.join("dataset", "samples")
        self.dir = {
            "r": root_dir,
            "p": os.path.join(root_dir, "problems"),
            "s": os.path.join(root_dir, "solutions"),
        }
        self.target, self.idx = None, 0
        self.orders = self._read_files(shuffle=shuffle, n_samples=n_samples, n_split=n_split, div=div)
        self.n_samples = len(self.orders)
        
    def _read_files(self, shuffle, n_samples, n_split, div=-1):
        # read file instances based on problem files
        files = ([f[2:] for f in os.listdir(self.dir['p']) if f[-3:] == 'txt'])    # e.g., p_0.txt -> 0.txt, p_999.txt -> 999.txt
        if shuffle:
            random.seed(0)  # fix an order of data file
            random.shuffle(files)
        if n_samples > 0 and len(files) >= n_samples:
            files = files[-n_samples:]

        # split dataset (div is splitting index)
        if div < 0:
            return files
        block = int(len(files) / n_split)
        files = files[block*div:block*(div+1)]
        return files

    def load_datasets(self):
        inputs, labels = {}, {}
        for f in self.orders:
            prob_f, sol_f = 'p_' + f, 's_' + f
            g = read_problem_from_file(path_prob=os.path.join(self.dir['p'], prob_f))
            g_ = read_solution_from_file(g=g, path_sol=os.path.join(self.dir['s'], sol_f))
            labels[f] = self._get_label_from_solution_graph(g_)
            inputs[f] = GridGraph(g=g, sol=labels[f])
        return inputs, labels

    def _get_label_from_solution_graph(self, g):
        from collections import defaultdict

        feasible_solutions = defaultdict(list)
        for edge in g.edges:
            if g.edges[edge]["tree"] == 0:
                continue
            net_id = g.edges[edge]["tree"]
            feasible_solutions[net_id].append(edge)
        return feasible_solutions

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.n_samples:
            self.idx = 0
            raise StopIteration
        self.idx += 1
        self.target = self.orders[self.idx - 1]

        prob_f, sol_f = 'p_' + self.target, 's_' + self.target
        g = read_problem_from_file(path_prob=os.path.join(self.dir['p'], prob_f))
        g_ = read_solution_from_file(g=g, path_sol=os.path.join(self.dir['s'], sol_f))
        return GridGraph(g=g, sol=self._get_label_from_solution_graph(g_))
