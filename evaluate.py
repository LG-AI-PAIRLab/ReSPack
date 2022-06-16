import os, time, re, json
from time import strftime
from multiprocessing import Pool
from functools import partial
from itertools import product
from common.graph import visualize, visualize_3d
from common.recorder import Recorder
from common.loader import Loader
from heuristic.heuristics import Heuristics
from MILP.multi_commodity import MILP
from rankingcost.rankingcost import RankingCost


MAX = float('inf')

def str2bool(st):
    return True if st in ["True", "true", "1", "t", "T"] else False

def get_stats(STs, iterations, duration, G):
    from itertools import chain
    time_st = duration / (iterations * G.n_nets)    # elapsed time to route a single Steiner Tree
    mean_terms = sum([ len(v) for v in G.netlists.values()]) / G.n_nets # averaged # of terminal nodes per Steiner Tree
    score = MAX if STs is None else len(set(chain(*list(zip(*STs))[0]))) # total edge sum
    label_solution_value = sum([len(v) for v in G.solution.values()])
    gain = MAX if (score < 0 or label_solution_value == 0) else score / label_solution_value
    return time_st, mean_terms, score, gain

def load_json(root_dir):
    configs = {}
    for cf in [f for f in os.listdir(root_dir) if f.endswith('json')]:
        with open(os.path.join('configs', cf), 'r') as file:
            configs[cf] = json.load(file)
            for col in ['shuffle', 'verbose', 'save']:
                configs[cf][col] = str2bool(configs[cf][col])
    return configs

def run(data_dir, sample_dir, log_dir, arg_algo, verbose, n_samples, save, shuffle, time_limits, n_split, iterables):
    div, sd = iterables
    if n_split >= 0:
        assert n_samples % n_split == 0
    

    if arg_algo == "heuristic":
        algo = Heuristics(**args)
        file_name = f"{sample_dir}_{arg_algo}_{args['route']}_{args['rrr']}_ins{n_samples}_seed{sd}_idx{div+1}"
    elif arg_algo == "milp":
        algo = MILP(**args)
        file_name = f"{sample_dir}_{arg_algo}_ins{n_samples}_seed{sd}_idx{div+1}"
    elif arg_algo == "rankingcost":
        algo = RankingCost(**args)
        file_name = f"{sample_dir}_{arg_algo}_ins{n_samples}_seed{sd}_idx{div+1}"
    else:
        raise NotImplementedError

    if time_limits:
        time_limits *= 60*60    # 1 hours = 60*60 s
        if arg_algo == 'milp':
            time_limits *= 1000 # ms
        algo.time_limits = time_limits
    algo.init_seed(sd)  # random seed

    loader = Loader(
        root_dir=os.path.join(data_dir, sample_dir), sample_dir=sample_dir, shuffle=shuffle, n_samples=n_samples, n_split=n_split, div=div,
    )
    print(file_name)
    r = Recorder(dirs=log_dir, samples=loader.orders, file_name=file_name, seeds=[sd])

    for idx_t, G in enumerate(loader):
        print(f'[{div+1}]   {idx_t+1} / {int(n_samples / n_split)}')
        sample = f"{arg_algo}_%05d" % int(re.findall(r"\d+", loader.target)[0])
        if verbose:
            print("ROUTING SAMPLE {s}.........................".format(s=sample))

        algo.init_graph(G)

        # run algorithm
        start = time.time()
        result = algo.run()
        duration = time.time() - start

        solutions, iterations = result['solution'], result['iteration']

        # compute statistics
        time_st, mean_terms, score, gain = get_stats(solutions, iterations, duration, G)

        r.record(
            sample=loader.target,
            iter=iterations,
            sol=str(solutions is not None),
            dur=duration,
            score=score,
            dur_per_st=time_st,
            mean_terms=mean_terms,
            gain=gain,
            seed=sd
        )
        r.save_as_file()

        if solutions is None:
            print("Route failed !")
            continue

        if save:
            save_dir = os.path.join('images', sample_dir)
            os.makedirs(save_dir, exist_ok=True)
            visualize(
                G,
                result["solution"],
                save=save,
                file_name="{f}_{sd}.png".format(f=os.path.join(save_dir, sample), sd=div),
            )
            if len(list(G)[-1]) == 3:
                visualize_3d(
                    G,
                    result["solution"],
                    save=save,
                    file_name="{f}_3d.png".format(f=os.path.join(save_dir, sample)),
                )

    

if __name__ == "__main__":
    configs = load_json('configs')
    for file_name, args in configs.items():
        for sample_dir in args['datasets']:
            dir_name = strftime("%y-%m-%d-%I:%M:%S",time.localtime())
            log_dir = os.path.join('logs', sample_dir, args['algo'], dir_name)
            os.makedirs(log_dir, exist_ok=True)

            with open(os.path.join(log_dir, 'config.json'), 'w') as f:
                json.dump(args, f)

            func = partial(
                run,
                args["data_dir"],
                sample_dir,
                log_dir,
                args["algo"],
                str2bool(args["verbose"]),
                args["n_samples"],
                args["save"],
                args["shuffle"],
                args["time_limits"],
                args["n_split"],
            )

            if args["n_process"] == 1:
                if args["n_seeds"] <= 0:
                    func([-1, 0])
                else:
                    for sd in range(args["n_seeds"]):
                        func([-1, sd])
            else:
                p = Pool(processes=args["n_process"])
                split_list = [-1] if args["n_split"] <= 0 else list(range(args["n_split"]))
                seeds = [0] if args["n_seeds"] <= 0 else list(range(args["n_seeds"]))
                p.map(func, product(split_list, seeds))
                p.close()
                p.join()

