import time
from ortools.linear_solver import pywraplp
from collections import defaultdict
from MILP.utils import make_roots, t_generator, extract_subgraph
from common.algorithm import Algorithm


class MILP(Algorithm):
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

        # initialize
        self.tsets = list(zip(*self.G.netlists.items()))[1]
        self.n_tsets = self.G.n_nets
        self.n_trials, self.min_obj, self.solution = 1, float("inf"), None

        # Randomly select roots
        if self.verbose:
            print(f"n_tsets: {self.n_tsets}, tlen {len([i for i in t_generator(self.tsets)])}")
        self.Rs = make_roots(self.tsets, self.n_trials)
        if self.verbose:
            print("make_roots finished...")

    def run(self):
        for R in self.Rs:
            solver = pywraplp.Solver.CreateSolver("SCIP")
            if self.verbose:
                print(f"solver created...")

            solver.SetSolverSpecificParametersAsString("randomization/randomseedshift = " + str(self.seed))

            # create a variable X
            X = [{} for _ in range(self.n_tsets)]
            for n in range(self.n_tsets):
                X[n] = {}
                for i, j in self.G.edges:
                    # variables for n-th terminal set, i and j nodes
                    # eq. (8) X is binary (0 or 1), bidirectional
                    X[n][(i, j)] = solver.IntVar(0, 1, "x[{0}][{1}][{2}]".format(n, i, j))
                    X[n][(j, i)] = solver.IntVar(0, 1, "x[{0}][{1}][{2}]".format(n, j, i))
            if self.verbose:
                print("X created...")
            # create the constraint
            # (i, j) in A
            for i, j in self.G.edges:
                if i > j:
                    continue  # avoid repetition. only consider i < j case
                # eq. (7) Sum_{n in N} x^n_ij + x^n_ji <= 1
                constraint = solver.Constraint(0, 1, "")
                for n in range(self.n_tsets):
                    constraint.SetCoefficient(X[n][(i, j)], 1)
                    constraint.SetCoefficient(X[n][(j, i)], 1)
            if self.verbose:
                print("X constraints created...")
            # create a variable Y
            # defaultdict can handle KeyError (because Y[r] doesn't exist)
            Y = defaultdict(defaultdict)
            # n is # of terminal sets
            # idx_t is for indexing y
            # t is t in T \ R
            for n, idx_t, t in t_generator(self.tsets):
                # we only consider t in T \ R
                # so, exclude "t == R[n]"
                if t == R[n]:
                    continue
                Y[idx_t] = {}
                # (i, j) in A
                for i, j in self.G.edges:
                    Y[idx_t][(i, j)] = solver.NumVar(0, 1, "y[{0}][{1}][{2}]".format(idx_t, i, j))
                    Y[idx_t][(j, i)] = solver.NumVar(0, 1, "y[{0}][{1}][{2}]".format(idx_t, j, i))
                    # eq. (6) 0 <= y^t_ij <= x^n_ij
                    solver.Add(Y[idx_t][(i, j)] - X[n][(i, j)] <= 0)
                    solver.Add(Y[idx_t][(j, i)] - X[n][(j, i)] <= 0)
                    solver.Add(Y[idx_t][(i, j)] >= 0)
                    solver.Add(Y[idx_t][(j, i)] >= 0)

                # create the constraint
                # eq. (5)
                for j in self.G.nodes:
                    # delta neg and pos are equal in undirected grid graph
                    # (i.e., adjacent nodes)
                    delta_neg = delta_pos = list(self.G[j])

                    # y value varies with j
                    if j == t:
                        cond = 1
                    elif j == R[n]:
                        cond = -1
                    else:
                        cond = 0

                    constraint = solver.Constraint(cond, cond, "")
                    for i in delta_neg:
                        constraint.SetCoefficient(Y[idx_t][(i, j)], 1)
                    for k in delta_pos:
                        constraint.SetCoefficient(Y[idx_t][(j, k)], -1)

            if self.verbose:
                print("Y created...")

            # create the constraint
            # eq. (9) node disjoint solution
            for j in self.G.nodes:
                delta_neg = list(self.G[j])
                cond = 0 if j in R else 1
                # Sum_n Sum_(i,j) x^n_ij <= cond ( cond = 0 if j in R, otherwise 1)
                constraint = solver.Constraint(-solver.infinity(), cond, "")
                for n in range(self.n_tsets):
                    for i in delta_neg:
                        constraint.SetCoefficient(X[n][(i, j)], 1)
            if self.verbose:
                print("node disjoint constraints created...")

            # create the objective
            objective = solver.Objective()
            for n in range(self.n_tsets):
                for i, j in self.G.edges:
                    objective.SetCoefficient(X[n][(i, j)], self.G[i][j]["weight"])
                    objective.SetCoefficient(X[n][(j, i)], self.G[j][i]["weight"])
            objective.SetMinimization()
            if self.verbose:
                print("objective created...")

            # setting time limits
            if self.time_limits:
                solver.SetTimeLimit(self.time_limits)

            if self.verbose:
                print(f"variables {len(solver.variables())}")
            status = solver.Solve()
            if self.verbose:
                print("solved...")

            # save best solution
            if (
                status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]
            ) and self.min_obj > objective.Value():  # better solution
                self.min_obj = objective.Value()
                self.solution = extract_subgraph(X)

            # exit if the solution is optimal
            if status == pywraplp.Solver.OPTIMAL:
                break
            self.n_trials += 1

        result = {
            "solution": self.solution,
            "iteration": self.n_trials,
        }
        return result
