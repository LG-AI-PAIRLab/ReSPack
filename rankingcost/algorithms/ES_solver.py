#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import copy
from itertools import chain
import pickle
import numpy as np
import networkx as nx
from rankingcost.utils import get_twopin_list, is_violate_constraints
from rankingcost.algorithms.ES_model import ESModel
from rankingcost.algorithms.ES_engine import EvolutionStrategy



class ESSolver:
    REWARD_SCALE = 20
    SIGMA = 0.1
    LEARNING_RATE = 0.01

    def __init__(
        self,
        train_num,
        POPULATION_SIZE=20,
        print_step=1,
        train_rank=True,
        model_save_path=None,
        logger=None,
        saver=None,
        verbose=None,
        time_limits=None,
    ):
        self.start_time = time.time()
        self.train_num = train_num
        self.POPULATION_SIZE = POPULATION_SIZE
        self.print_step = print_step
        self.train_rank = train_rank
        self.verbose = verbose
        self.model_save_path = model_save_path if model_save_path else "./model.pkl"
        self.time_limits = time_limits
        self.best_step = 5

        self.logger = logger
        self.saver = saver
        self.es = EvolutionStrategy(
            self.get_reward,
            self.log_function,
            self.stop_function,
            self.saver,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
            num_threads=self.POPULATION_SIZE,
            verbose=self.verbose,
        )

    def solve(self, G):
        # multipin to twopin list
        net_num = len(G.netlists)
        pair_num = 0
        sg_pairs_per_net = []
        constraints = []
        for i in range(len(G.netlists)):
            sg_pairs = get_twopin_list(G.netlists[i + 1])
            sg_pairs_per_net.append(sg_pairs)
            pair_num += len(sg_pairs)
            assert all([len(sg) == 2 for sg in sg_pairs])
            if 'constraints' in G.graph.keys() and G.graph['constraints']:
                constraints.append(G.graph['constraints'][i + 1])
            else:
                constraints.append({})

        if self.verbose:
            print(f"Total T={net_num}, pair={pair_num}")

        self.mini_cost = 1e6
        self.max_cost = len(G)

        self.best_solution = None
        self.best_traces = None
        self.best_reward = -1e6
        self.best_reward_step = 0

        # init model & es
        self.model = ESModel(net_num=net_num, pair_num=pair_num, node_num=len(G))
        self.es.reset()
        self.es.set_weights(self.model.get_weights())

        self.maze_data = G
        self.sg_pairs_per_net = sg_pairs_per_net
        self.constraints = constraints

        # run es
        self.es.run(iterations=self.train_num)

        if self.saver:
            print("save final model!")
            self.saver.save(self.es.get_weights())

    def close(self):
        self.es.close()

    def log_function(self, iteration, weights, reward, rewards, time_duration):
        if self.logger:
            self.logger.update_data({"reward": reward, "rewards": rewards, "time": time_duration, "step": iteration})
            self.logger.display_info()
            self.logger.plot_figure()
            self.logger.save_solution(self)

    def stop_function(self, iteration, weights, reward, rewards, time_duration):
        self.iteration = iteration
        if time_duration >= self.time_limits:
            return True
        if self.best_reward < reward:
            self.best_reward = reward
            self.best_reward_step = iteration
            self.best_solution = self.net_rank_now
            self.best_traces = self.traces_now
            self.best_iteration = iteration
            return False
        if self.best_reward == reward and reward == -self.REWARD_SCALE:
            return False
        if self.best_reward >= reward and (iteration - self.best_reward_step) >= self.best_step:
            return True

        return False

    def get_reward(self, weights):
        cost_now = self.get_solution(weights)[0]
        if cost_now is None or cost_now >= self.max_cost:
            cost_now = self.max_cost
        cost_now = cost_now / self.max_cost
        reward_now = -cost_now
        return reward_now * self.REWARD_SCALE

    def get_solution(self, weights):
        self.model.set_weights(weights)
        net_rank, maze_weights = self.get_maze_weights()

        cost_now, traces_now = self.planOnSolutionNx(
            net_rank, maze_weights, self.maze_data, self.sg_pairs_per_net
        )
        self.cost_now = cost_now
        self.traces_now = traces_now
        self.net_rank_now = net_rank
        return cost_now, traces_now, net_rank

    def get_maze_weights(self):
        prediction = self.model.get_rank_prob()
        solution_rank_per_net = []
        solution_rank = []
        if self.train_rank:
            net_rank = np.argsort(prediction)[::-1]
            idx = 0
            for sg_pairs in self.sg_pairs_per_net:
                sorted_rank = list(range(len(sg_pairs)))
                solution_rank_per_net.append(sorted_rank)
                solution_rank.extend(list(np.array(sorted_rank) + idx))
                idx += len(sg_pairs)
        else:
            net_rank = range(len(prediction))
            for sg_pairs in self.sg_pairs_per_net:
                solution_rank_per_net.append(list(range(len(sg_pairs))))
            solution_rank = range(len(maze_now))

        maze_now = copy.copy(self.model.get_maze_data())

        maze_now = maze_now[solution_rank]
        return net_rank, maze_now

    def planOnSolutionNx(self, net_rank_now, maze_weights, maze_data, sg_pairs_per_net):
        net_rank_now = list(net_rank_now)
        cost_all = 0
        new_sg_pairs_per_net = []
        new_terminals_per_net = []
        new_const_per_net = []
        for t in net_rank_now:
            new_sg_pairs = []
            new_terminals = []
            for i in range(len(sg_pairs_per_net[t])):
                new_sg_pairs.append(sg_pairs_per_net[t][i])
                new_terminals.append(sg_pairs_per_net[t][i][0])
                new_terminals.append(sg_pairs_per_net[t][i][1])
            new_sg_pairs_per_net.append(new_sg_pairs)
            new_terminals_per_net.append(list(set(new_terminals)))
            new_const_per_net.append(self.constraints[t])

        g = copy.deepcopy(self.maze_data)

        traces = []
        traces_edge = []

        node_index = {}
        for j, v in enumerate(g.nodes()):
            node_index[v] = j

        idx = 0
        extra_costs = [None] * len(maze_weights)
        extra_costs[-1] = np.zeros_like(maze_weights[0])
        for i in reversed(range(len(maze_weights)-1)):
            extra_costs[i] = extra_costs[i+1] + maze_weights[i+1]

        for t, new_sg_pairs in enumerate(new_sg_pairs_per_net):
            trace_per_net_now = []
            trace_edge_per_net_now = []
            remain_terminals = list(chain(*new_terminals_per_net[t+1:]))
            g.remove_nodes_from(remain_terminals)
            for i, sg_pair in enumerate(new_sg_pairs):
                if time.time() - self.start_time >= self.time_limits:
                    return None, None
                try:
                    get_extra_cost = lambda u, v, w: extra_costs[idx][node_index[v]]
                    trace_now = nx.astar_path(g, sg_pair[0], sg_pair[1], heuristic=dist, weight=get_extra_cost)
                    trace_edge_now = list(zip(trace_now[:-1], trace_now[1:]))
                    idx += 1
                except Exception as e:
                    # impossible to find path
                    if self.verbose:
                        print(f'[{idx}] T{net_rank_now.index(t)}, {e}')
                    return None, None
                trace_per_net_now.extend(trace_now)
                trace_edge_per_net_now.extend(trace_edge_now)
                if len(trace_edge_now) == 0:
                    return None, None

            trace_per_net_now = list(set(trace_per_net_now))
            trace_edge_per_net_now = list(set(trace_edge_per_net_now))
            new_st = (trace_edge_per_net_now, trace_per_net_now)
            is_violated, const_info = \
                is_violate_constraints(self.maze_data, [], new_st, new_terminals_per_net[t], new_const_per_net[t])
            if is_violated:
                if self.verbose:
                    print(f'[{idx}] T{net_rank_now.index(t)} CONST VIOLATE: {const_info}')
                return None, None

            cost_per_net = len(trace_edge_per_net_now)
            
            g.remove_nodes_from(trace_per_net_now)  # next net should not conjest partial solution
            g.add_nodes_from(remain_terminals)
            g.add_edges_from([(u, v) for u in remain_terminals for v in self.maze_data[u] if v in g])
            
            if cost_per_net == 0:
                return None, None
            traces.append(trace_per_net_now)
            traces_edge.append(trace_edge_per_net_now)

            cost_all += cost_per_net

        return cost_all, traces_edge

    def load(self, filename="weights.pkl"):
        with open(filename, "rb") as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()


def dist(a, b):
    return np.sum((np.array(a) - np.array(b)) ** 2) ** 0.5
