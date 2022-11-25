# ReSPack: A Large-Scale Rectilinear Steiner Tree Packing Data Generator and Benchmark
This repository contains the codebase of the paper  
[ReSPack: A Large-Scale Rectilinear Steiner Tree Packing Data Generator and Benchmark](https://openreview.net/forum?id=TpYf4EumAi)  
by Kanghoon Lee*, Youngjoon Park*, Han-Seul Jeong*, Sunghoon Hong, Deunsol Yoon, Sungryull Sohn, Minu Kim, Hanbum Ko, Moontae Lee, Honglak Lee, Kyunghoon Kim, Euihyuk Kim, Seonggeon Cho, Jaesang Min, Woohyung Lim

Cite:
```latex
@inproceedings{
lee2022respack,
title={Re{SP}ack: A Large-Scale Rectilinear Steiner Tree Packing Data Generator and Benchmark},
author={Kanghoon Lee and Youngjoon Park and Han-Seul Jeong and Deunsol Yoon and Sunghoon Hong and Sungryull Sohn and Minu Kim and Hanbum Ko and Moontae Lee and Honglak Lee and Kyunghoon Kim and Euihyuk Kim and Seonggeon Cho and Jaesang Min and Woohyung Lim},
booktitle={NeurIPS 2022 Workshop on Synthetic Data for Empowering ML Research},
year={2022},
url={https://openreview.net/forum?id=TpYf4EumAi}
}
```

# Summary
<table style="text-align:center; margin:auto;">
  <tr>
    <th style="text-align:center; margin:auto;">Real PCB</th>
    <th style="text-align:center; margin:auto;">RSTPP in Real PCB</th>
    <th style="text-align:center; margin:auto;">RSTPP in ReSPack</th>
  </tr>
  <tr>
    <td><img src="asset/realpcb_sample1.jpg" width="220" height="220"/></td>
    <td><img src="asset/realpcb_rstpp_sample1.jpg" width="220" height="220"/></td>
    <td><img src="asset/respack_rstpp_sample1.jpg" width="220" height="220"/></td>
  </tr>
</table>

Combinatorial optimization (CO) has been studied as a useful tool for modeling industrial problems, but it still remains a challenge in complex domains because of the NP-hardness.
With recent advances in machine learning, the field of CO is shifting to the study of neural combinatorial optimization using a large amount of data, showing promising results in some CO problems.
Rectilinear Steiner tree packing problem (RSTPP) is a well-known CO problem and is widely used in modeling wiring problem among components in a printed circuit board and an integrated circuit design.
Despite the importance of its application, the lack of available data has restricted to fully leverage machine learning approaches.
In this paper, we present ReSPack, a large-scale synthetic RSTPP data generator and a benchmark.
ReSPack includes a source code for generating RSTPP instances of various types with different sizes, test instances generated for the benchmark evaluation, and implementations of several baseline algorithms.

You can download ReSPack dataset [here](http://dx.doi.org/10.17605/OSF.IO/BJ9R3).

-------------

## Usage

### Setup
```
pip install -r requirements.txt
```

### Example: Generate single instance
```
from stppgen.stpsolution import GridSTP2opt, GridSTPfast
stp1 = GridSTP2opt(dim=[2, 10, 10], max_ntrees=2, max_nterminals=[5, 5])
stp1.generate()
stp1.save_problem_txt('sample1_prob.txt')
stp1.save_solution_txt('sample1_sol.txt')
stp1.draw_graph(5, 5, save='sample1.png')

stp2 = GridSTPfast(dim=[2, 10, 10], max_ntrees=2, max_nterminals=[5, 5])
stp2.generate()
stp1.save_problem_txt('sample1_prob.txt')
stp1.save_solution_txt('sample1_sol.txt')
stp2.draw_graph(5, 5, save='sample2.png')
```


### Example: Generate multiple instances with multiprocessing
```
from stppgen.utils.distribution import Multinomial, Poisson
from stppgen.generator import DataGenerator
dist1 = Multinomial(vmin=3, vmax=10)
dist2 = Poisson(mu=2)
gen = DataGenerator(n=10, method='fast', dim=[1, 10, 10],
                    ntrees=dist1, nterminals=dist2,
                    save_to_dir='samples/', save_plot=True, overwrite=True)
gen.run(ncores=4)
```

### Example: read generated data
```
from stppgen.utils.graphio import (read_problem_from_file, read_solution_from_file, 
draw_graph_3d)

i = 1
path_prob = f"samples/problems/p_{i}.txt"
path_sol = f"samples/solutions/s_{i}.txt"

g = read_problem_from_file(path_prob)
g_sol = read_solution_from_file(g, path_sol)

draw_graph_3d(g_sol, 10, 10, 1, save=f'test.png')
```


### Generate dataset with config file
Prepare your config file for data genaration. 
Sample config files are provided in `dataset_config` (default root directory). 
Then, run the following command.
```
python generate.py --root [ROOT_DIRECTORY] --name [DATASET_NAME] --n_samples [N_SAMPLES] --n_cores [N_CORES] [--init_base_dir [INIT_BASE_DIR]]
```



### Algorithm template
1. Create custom algorithm class
In `custom/custom.py`
```
from common.algorithm import Algorithm

class CustomAlgorithm(Algorithm):
    def __init__(self, **args):
        super().__init__(args)
        # initialize
    
    def run(self):
        ST = STEINER_TREES(=[(EDGE_LIST, NODE_LIST), ...]) if Success else None
        iteration = Number of iteration if exists else 1
        result = {"solution": ST, "iteration": iteration}
        return result
```

2. Add the following code segment 
In `evaluate.py`, `run()` function,
```
from custom.custom import CustomAlgorithm

if arg_algo == "custom":
    algo = CustomAlgorithm(**args)
    file_name = f"{sample_dir}_{arg_algo}_ins{n_samples}_seed{sd}_idx{div+1}"
```

### Evaluate algorithm
Prepare your config file which contains arguments for evaluating algorithm. 
Sample config files are provided in `configs`. 
Then, run the following command.
```
python evaluate.py
```


## Directory
```
ReSPack  
├── dataset_config: config files for dataset generation  
│   |── UC  
│   |── NWA  
│   └── NWA_LS_WL  
│   
├── stppgen: RSTPP generator  
│   └── utils  
│  
├── configs: config files for evaluation  
│  
├── common: common class  
│  
├── MILP: MILP method (OR-Tools)  
│  
├── heuristic: heuristic method (sequential routing)  
│  
├── rankingcost: evolution strategy based routing  
│   └── algorithms: core  
│  
├── generate.py: generate dataset  
└── evaluate.py: evaluate algorithm
```

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
