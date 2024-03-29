| key                       | value                                       | description                                                                                                                                     |
|---------------------------|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| "description"             | "NWA_LS_WL"                                 | Description for the dataset                                                                                                                     |
| "dim"                     | [2, 8, 8]                                   | Grid size of [z, x, y]                                                                                                                          |
| "method"                  | "2opt_obsconst"                             | Data generation method {2opt/fast}_{constraints} <br>see stppgen>generator.py                                                                   |
| "distribution_ntrees"     | {"type": "multinomial", "min": 4, "max": 5} | Sample # of trees from multinomial distribution where "min" <= N < "max"                                                                        |
| "distribution_nterminals" | {"type": "poisson", "mu": 2}                | Sample # of terminal nodes from the Poisson distribution                                                                                        |
| "r_min"                   | 3                                           | The minimum distanance between terminal nodes in a Steiner tree                                                                                 |
| "r_max"                   | 8                                           | The maximum distanance between the first terminal node and other terminal nodes in a Steiner tree                                               |
| "obstacle"                | 4                                           | Scale of the large obstacle size for NWA constraints. See stppgen>constraints.py>ObstacleAssigner                                               |
| "margin_max"              | 1                                           | The maximum size of margin for LS contraints. The margin size is sampled from [1, margin_max]. See stppgen>constraints.py>TreeMarginConstraints |
| "random_seed"             | 1234567                                     |                                                                                                                                                 |