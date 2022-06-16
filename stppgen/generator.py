import os
from stppgen.base import BaseDataGenerator
from stppgen.stpsolution import * 
from stppgen.constrainedstp import *
from stppgen.sequentialstp import *


class DataGenerator(BaseDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def init_stp_generator(self, dim, max_ntrees, max_nterminals, idx=None):
        if self.method == 'fast':
            stp = GridSTPfast(dim, max_ntrees, max_nterminals, 
                              r_min=self.r_min, r_max=self.r_max, 
                              **self.kwargs)
        elif self.method == '2opt':
            stp = GridSTP2opt(dim, max_ntrees, max_nterminals, 
                              r_min=self.r_min, r_max=self.r_max, 
                              **self.kwargs)
        elif self.method == '2opt_obs':
            stp = GridSTP2optObstacle(dim, max_ntrees, max_nterminals, 
                                      r_min=self.r_min, r_max=self.r_max, 
                                      **self.kwargs)
        elif self.method == 'fast_obs':
            stp = GridSTPfastObstacle(dim, max_ntrees, max_nterminals, 
                                      r_min=self.r_min, r_max=self.r_max, 
                                      **self.kwargs)
        elif self.method == '2opt_obsconst':
            stp = GridSTP2optObstacleandCostraint(dim, max_ntrees, max_nterminals, 
                                                  r_min=self.r_min, r_max=self.r_max, 
                                                  idx=idx, **self.kwargs)
        elif self.method == 'fast_obsconst':
            stp = GridSTPfastObstacleandCostraint(dim, max_ntrees, max_nterminals, 
                                                  r_min=self.r_min, r_max=self.r_max, 
                                                  idx=idx, **self.kwargs)
        else:
            raise NotImplementedError
        return stp
    

class SequentialDataGenerator(BaseDataGenerator):
    def __init__(self, *args, **kwargs):
        self.seed_files = kwargs['seed_files']
        self.nsamples_each = kwargs['n']
        self.random_seed = kwargs['random_seed']
        self.overwrite = kwargs['overwrite']
        self.save_to_dir = kwargs['save_to_dir']
        self.margin = kwargs['margin']
        self.save_plot = kwargs['save_plot']
        self.method = kwargs['method']
        self.kwargs = kwargs
    
    def init_stp_generator(self, seed_problem, seed_solution):
        if self.method == 'faster':
            stp = SequentialGenerator(path_problem=seed_problem, path_solution=seed_solution,
                              **self.kwargs)
        elif self.method == 'faster_obs':
            stp = SequentialGridSTPfastObstacle(path_problem=seed_problem, path_solution=seed_solution,
                              **self.kwargs)
        elif self.method == 'faster_obsconst':
            stp = ConstrainedSequentialGridSTPfast(path_problem=seed_problem, path_solution=seed_solution,
                              **self.kwargs)
        else:
            raise NotImplementedError
        return stp
    
    def generate_instance(self, seed_problem, seed_solution):
        stp = self.init_stp_generator(seed_problem, seed_solution)
        filename_prob = seed_problem.split('/')[-1].split('.')[0]
        filename_sol = seed_solution.split('/')[-1].split('.')[0]
        for idx in range(self.nsamples_each):
            if self.random_seed is not None:
                random_seed = self.random_seed + idx
            else:
                random_seed = None
            self.kwargs['random_seed'] = random_seed
            path_for_problem = os.path.join(self.save_to_dir, f'problems/{filename_prob}_{idx}.txt')
            path_for_solution = os.path.join(self.save_to_dir, f'solutions/{filename_sol}_{idx}.txt')

            if not os.path.exists(path_for_solution) or self.overwrite:
                stp.generate()
                stp.save_problem_txt(path_for_problem)
                stp.save_solution_txt(path_for_solution)
                if self.save_plot:
                    filename = filename_sol.replace('s', 'f')
                    path_for_figure = os.path.join(self.save_to_dir, f'solutions/{filename}_{idx}.png')
                    stp.draw_graph(5, 5, save=path_for_figure)        
    
    def run(self, ncores=1):
        if self.save_to_dir is not None:
            if not os.path.exists(self.save_to_dir):
                os.makedirs(self.save_to_dir)
            os.makedirs(os.path.join(self.save_to_dir, 'problems'), exist_ok=self.overwrite)
            os.makedirs(os.path.join(self.save_to_dir, 'solutions'), exist_ok=self.overwrite)

        if ncores > 1:
            pool = mp.Pool(len(self.seed_files))
            jobs = []
            for i, (p, s) in enumerate(self.seed_files):
                jobs.append(pool.apply_async(self.generate_instance, (p, s)))
            for j in jobs:
                j.get()
            pool.close()
            pool.join()
        else:
            for i, (p, s) in enumerate(self.seed_files):
                self.generate_instance(p, s)
        