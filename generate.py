import os
import json
from argparse import ArgumentParser
from stppgen.generator import DataGenerator
from stppgen.utils.distribution import Multinomial, Poisson


if __name__ == "__main__":
    parser = ArgumentParser('Argument for dataset generation.')
    parser.add_argument('--dest', type=str, default='respack', help='destination to save datasets.')
    parser.add_argument('--config_root', type=str, default='dataset_configs', help='root directory of config files.')
    parser.add_argument('--name', type=str, default='UC', help='directory name containing target config files.')
    parser.add_argument('--init_base_dir', type=str, default=None, help='directory name containing base datasets which shares NWA with.')
    parser.add_argument('--n_samples', type=int, default=100, help='total number of samples to generate.')
    parser.add_argument('--n_cores', type=int, default=10, help='total number of cpu cores for multiprocessing')
    
    args = parser.parse_args()

    dir_config = os.path.join(args.config_root, args.name)
    
    files = os.listdir(dir_config)
    files.sort()
    for fp in files:
        filepath = os.path.join(dir_config, fp)
        with open(filepath, 'r') as f:
            config = json.load(f)

        if config['distribution_ntrees']['type'] == 'multinomial':
            dist_ntree = Multinomial(vmin=config['distribution_ntrees']['min'],
                                     vmax=config['distribution_ntrees']['max'])

        if config['distribution_nterminals']['type'] == 'multinomial':
            dist_nterminals = Multinomial(vmin=config['distribution_nterminals']['min'],
                                          vmax=config['distribution_nterminals']['max'])
        elif config['distribution_nterminals']['type'] == 'poisson':
            dist_nterminals = Poisson(mu=config['distribution_nterminals']['mu'])
        else:
            raise NotImplementedError
        
        save_plot = False
        target_dir = f"{args.dest}/{config['description']}_{config['dim'][-1]}/"
        
        init_base_dir = args.init_base_dir
        if init_base_dir and 'obs' in config['method'] and 'const' in config['method']:
            init_base_dir = os.path.join(args.dest, f"{args.init_base_dir}_{config['dim'][-1]}")
            if os.path.exists(init_base_dir):
                print(f'Init base dir exists. NWA will be shared with {init_base_dir}')
            else:
                init_base_dir = None
                print(f'Init base dir does not exists. NWA will be initialized from scratch.')        
        
        gen = DataGenerator(n=args.n_samples, ntrees=dist_ntree, nterminals=dist_nterminals, 
                            save_to_dir=target_dir, save_plot=save_plot, overwrite=True, 
                            init_base_dir=init_base_dir,
                            **config)
        gen.run(ncores=args.n_cores, start=0)
        print('.', end='')

