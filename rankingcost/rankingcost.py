import random
import numpy as np
from common.algorithm import Algorithm
from rankingcost.algorithms.ES_solver import ESSolver as Solver
from rankingcost.utils import extract_subgraph


class RankingCost(Algorithm):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.train_rank = True

    def run(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

        solver = Solver(
            POPULATION_SIZE=self.n_particles,
            train_num=self.n_trials,
            verbose=self.verbose,
            time_limits=self.time_limits,
        )
        solver.solve(self.G)
        ST = extract_subgraph(solver)
        if ST:
            iteration = solver.best_iteration + 1 
        else:
            iteration = solver.iteration + 1
        
        result = {"solution": ST, "iteration": iteration}
        return result
