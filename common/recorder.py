import pandas as pd
import os
class Recorder:
    def __init__(self, samples,  grid=(-1,-1), dirs='logs', file_name='samples', seeds=[-1]):
        self.samples = samples
        self.grid = grid    # (n, m)
        self.seeds = seeds

        # sample dictionary for each seed.
        self.iterations = {sd : {s:-1 for s in samples} for sd in seeds}
        self.solved = {sd:{s:-1 for s in samples} for sd in seeds}
        self.duration = {sd:{s:-1 for s in samples} for sd in seeds}
        self.scores = {sd:{s:-1 for s in samples} for sd in seeds}
        self.duration_per_steiner_tree = {sd:{s:-1 for s in samples} for sd in seeds}
        self.n_terminals = {sd:{s:-1 for s in samples} for sd in seeds}
        self.gains = {sd:{s:-1 for s in samples} for sd in seeds}
        self.dirs, self.file_name = dirs, file_name

    def record(self, sample, iter, sol, dur, score, dur_per_st, mean_terms, gain, seed=-1):
        self.iterations[seed][sample] = iter
        self.solved[seed][sample] = sol
        self.duration[seed][sample] = dur
        self.scores[seed][sample] = score
        self.duration_per_steiner_tree[seed][sample] = dur_per_st
        self.n_terminals[seed][sample] = mean_terms
        self.gains[seed][sample] = gain

    def save_as_file(self):
        data = [ [sd, s, self.iterations[sd][s], self.solved[sd][s], self.gains[sd][s], self.duration[sd][s],
         self.scores[sd][s], self.duration_per_steiner_tree[sd][s], self.n_terminals[sd][s]]
         for s in self.samples for sd in self.seeds]

        file_path = os.path.join(self.dirs, self.file_name + '.csv')
        if not os.path.exists(self.dirs):
            os.mkdir(self.dirs)
            
        records = pd.DataFrame(data)
        records.columns = ['seed', 'instance', 'iterations', 'solved', 'gain', 'time', 'score','time_per_ST', 'n_terminals']

        records.to_csv(file_path, index=False)
        print(file_path, ' is saved !')

