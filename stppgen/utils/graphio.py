import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
from scipy.spatial import ConvexHull


def parse_meta(lines):
    dim, n_trees, n_terminals = None, None, None
    for l in lines:
        if dim and n_trees and n_terminals:
            break
        if 'grid size' in l:
            l = l.replace('\n', '')
            l = l.split(':')[-1]
            l = l.strip()
            dim = eval(l)
        if 'number of trees' in l:
            l = l.replace('\n', '')
            l = l.split(':')[-1]
            l = l.strip()
            n_trees = eval(l)
        if 'number of terminals' in l:
            l = l.replace('\n', '')
            l = l.split(':')[-1]
            l = l.strip()
            n_trees = eval(l)
    meta = {'dim': dim, 'n_trees': n_trees, 'n_terminals': n_terminals}
    return meta
            

def parse_edges(lines):
    edges = []
    is_append_to_edges = False
    for l in lines:
        # parse edges list
        if "GRAPH EDGES BEGIN" in l:
            is_append_to_edges = True
        elif "GRAPH EDGES END" in l:
            break
        else:
            if is_append_to_edges:
                l = l.replace('\n', '')
                l = l.split(' - ')
                e = [eval(x) for x in l]
                edges.append(e)
    return edges


def parse_terminals(lines):
    terminals = {}
    is_append_to_terminals = False
    id = 1
    for l in lines:
        if f"TERMINAL NODES of TREE {id} BEGIN" in l:
            is_append_to_terminals = True
            terminals[id] = []
        elif f"TERMINAL NODES of TREE {id} END" in l:
            is_append_to_terminals = False
            id += 1
        elif '[TERMINAL NODES END]' in l:
            break
        else:
            if is_append_to_terminals:
                try:
                    terminals[id].append(eval(l))
                except SyntaxError:
                    pass
    return terminals


def parse_constraints(lines):
    constraints = {}
    is_append_to_constr = False
    id = 1
    for l in lines:
        if f"STEINER TREE {id} CONSTRAINTS BEGIN" in l:
            is_append_to_constr = True
            constraints[id] = {}
        elif f"STEINER TREE {id} CONSTRAINTS END" in l:
            is_append_to_constr = False
            id += 1
        elif '[CONSTRAINTS END]' in l:
            break
        else:
            if is_append_to_constr:
                l = l.strip().split(": ")
                constraints[id][l[0]] = eval(l[1])
    return constraints


def parse_trees(lines):
    trees = {}
    is_append_to_trees = False
    id = 1
    for l in lines:
        if f"[STEINER TREE {id} BEGIN]" in l:
            is_append_to_trees = True
            trees[id] = []
        elif f"[STEINER TREE {id} END]" in l:
            is_append_to_trees = False
            id += 1
        elif 'EOF' in l:
            break
        else:
            if is_append_to_trees:
                l = l.replace('\n', '')
                l = l.split(' - ')
                try:
                    e = [eval(x) for x in l]
                except SyntaxError:
                    pass
                else:
                    trees[id].append(e)
    return trees


def read_problem_from_file(path_prob):
    with open(path_prob) as f:
        lines = f.readlines()

    meta = parse_meta(lines)
    edges = parse_edges(lines)
    terminals = parse_terminals(lines)
    constraints = parse_constraints(lines)

    g = nx.from_edgelist(edges)
    g.dim = meta['dim']
    
    # init graph
    for t in g.nodes:
        g.nodes[t]['is_terminal'] = False
        g.nodes[t]['tree'] = 0  # 0 is background
    for e in g.edges:
        g.edges[e]['tree'] = 0  # 0 is background
        g.edges[e]['weight'] = 1  # 0 is background

    # set terminals
    for k, ts in terminals.items():
        for t in ts:
            g.nodes[t]['is_terminal'] = True
            g.nodes[t]['tree'] = k
    # set constraints
    g.graph = {'constraints': constraints}
    return g


def read_solution_from_file(g, path_sol):
    with open(path_sol) as f:
        lines = f.readlines()
    trees = parse_trees(lines)
    # set trees
    for k, es in trees.items():
        # update edge
        for e in es:
            g.edges[e]['tree'] = k
        # update node
        nodes = set(itertools.chain.from_iterable(es))
        for n in nodes:
            g.nodes[n]['tree'] = k
        # update connectivity
        for e in g.edges(nodes):
            g.edges[e]['weight'] = np.inf
    return g


def draw_graph_2d(g, x, y, save, color_plot=False, hull=False):
    n_tree = max([g.nodes[x]['tree'] for x in g.nodes])
    pos = dict((n, n) for n in g.nodes())
    axes = (plt.figure(constrained_layout=True, figsize=(x, y)).subplots(1, 1, sharex=True, sharey=True))
    if color_plot:
        cmap = plt.cm.get_cmap('rainbow', n_tree + 1)
        for n in g.nodes:
            i = g.nodes[n]['tree']
            col = cmap(i) if i > 0 else 'black'
            alpha = 1 if g.nodes[n]['tree'] > 0 else 0.5
            shape = 'o'
            size = 100 if g.nodes[n]['is_terminal'] else 10
            nx.draw_networkx_nodes(g, pos=pos, nodelist=[n], ax=axes,
                                   node_color=[col], node_shape=shape, alpha=alpha, node_size=size)
        for e in g.edges:
            i = g.edges[e]['tree']
            if i > 0:
                col = cmap(i)
                nx.draw_networkx_edges(g, pos=pos, edgelist=[e], edge_color=col)
        if hull:
            for i in range(1, n_tree + 1):
                col = cmap(i)
                pos_terminal = np.array([n for n in g.nodes() if g.nodes[n]['tree'] == i])
                if len(pos_terminal) > 2:
                    cvxhull = ConvexHull(pos_terminal)
                    plt.fill(pos_terminal[cvxhull.vertices, 0],
                            pos_terminal[cvxhull.vertices, 1], color=col, alpha=0.2)
    else:
        col = 'black'
        for n in g.nodes:
            alpha = 0.8 if g.nodes[n]['tree'] > 0 else 0.5
            shape = 'o'
            size = 100 if g.nodes[n]['is_terminal'] else 10
            nx.draw_networkx_nodes(g, pos=pos, nodelist=[n], ax=axes,
                                   node_color=[col], node_shape=shape, alpha=alpha, node_size=size)
        for e in g.edges:
            i = g.edges[e]['tree']
            if i > 0:
                nx.draw_networkx_edges(g, pos=pos, edgelist=[e], ax=axes, edge_color=col)

    if save is not None:
        plt.savefig(save, format="PNG")

    plt.show()
    plt.close()


def draw_graph_3d(g, x, y, z, save, color=False):
    axes = (plt.figure(constrained_layout=True,
                       figsize=(x * z, y)).subplots(1, z, sharex=True, sharey=True))
    n_tree = max([g.nodes[x]['tree'] for x in g.nodes])
    if color:
        cmap = plt.cm.get_cmap('rainbow', n_tree + 1)

    if not isinstance(axes, Iterable):
        axes = [axes]

    for k, ax in enumerate(axes):
        pos = dict((n, n[0:2]) for n in g.nodes)
        
        if z > 1:
            nodelist = [n for n in g.nodes if n[-1] == k]
            edgelist = [e for e in g.edges if (e[0][-1] == k and e[1][-1] == k) and g.edges[e]['tree'] > 0]
        else:
            nodelist = [n for n in g.nodes]
            edgelist = [e for e in g.edges if g.edges[e]['tree'] > 0]

        if color:
            edgecolor = [cmap(g.edges[e]['tree']) if g.edges[e]['tree'] > 0 else 'black' for e in edgelist]
            node_color = [cmap(g.nodes[n]['tree']) if g.nodes[n]['is_terminal'] else 'grey' for n in nodelist]
        else:
            edgecolor = 'black'
            node_color = ['black' if g.nodes[n]['is_terminal'] else 'grey' for n in nodelist]
        node_size = [30 if g.nodes[n]['is_terminal'] else
                     5 if g.nodes[n]['tree'] > 0 else 5 for n in nodelist]

        nx.draw(g, pos, ax=ax, nodelist=nodelist, edgelist=edgelist,
                with_labels=False, edge_color=edgecolor, width=2,
                node_color=node_color, node_size=node_size)

    if save is not None:
        plt.savefig(save, format="PNG")
    plt.show()
    plt.close()
    
    
def draw_graph_3d_ver2(g, x, y, z, save, color=False):
    n_tree = max([g.nodes[x]['tree'] for x in g.nodes])
    if color:
        cmap = plt.cm.get_cmap('rainbow', n_tree + 1)

    for k, _ in enumerate(range(z)):
        axes = (plt.figure(constrained_layout=True,
                       figsize=(x * 1, y)).subplots(1, 1, sharex=True, sharey=True))
        pos = dict((n, n[0:2]) for n in g.nodes)
        
        if z > 1:
            nodelist = [n for n in g.nodes if n[-1] == k]
            edgelist = [e for e in g.edges if (e[0][-1] == k and e[1][-1] == k) and g.edges[e]['tree'] > 0]
        else:
            nodelist = [n for n in g.nodes]
            edgelist = [e for e in g.edges if g.edges[e]['tree'] > 0]

        if color:
            edgecolor = [cmap(g.edges[e]['tree']) if g.edges[e]['tree'] > 0 else 'black' for e in edgelist]
            node_color = [cmap(g.nodes[n]['tree']) if g.nodes[n]['is_terminal'] else 'grey' for n in nodelist]
        else:
            edgecolor = 'black'
            node_color = ['black' if g.nodes[n]['is_terminal'] else 'grey' for n in nodelist]
        node_size = [30 if g.nodes[n]['is_terminal'] else
                     5 if g.nodes[n]['tree'] > 0 else 5 for n in nodelist]

        nx.draw(g, pos, ax=axes, nodelist=nodelist, edgelist=edgelist,
                with_labels=False, edge_color=edgecolor, width=2,
                node_color=node_color, node_size=node_size)

        if save is not None:
            plt.savefig(save.replace(".png", f"_{k}.png"), format="PNG")
        plt.show()
        plt.close()
    


def draw_graph_1layer(g, x, y, z, save, color=False):
    layer = z
    axes = (plt.figure(constrained_layout=True,
                    figsize=(x * 1, y)).subplots(1, 1, sharex=True, sharey=True))
    n_tree = max([g.nodes[x]['tree'] for x in g.nodes])
    layer = z
    for k in range(layer):
        pos = dict((n, n[:2]) for n in g.nodes)
        nodelist = [n for n in g.nodes if n[-1] == k]
        edgelist = [e for e in g.edges if (e[0][-1] == k and e[1][-1] == k) and g.edges[e]['tree'] > 0]
        edgecolor = 'tab:red' if k == 0 else 'tab:green'
        node_color = ['black' if g.nodes[n]['is_terminal'] else 'grey' for n in nodelist]
        size_t, size_n = 2, 0.001
        node_size = [size_t if g.nodes[n]['is_terminal'] else
                    size_n if g.nodes[n]['tree'] > 0 else size_n for n in nodelist]
        nx.draw(g, pos, ax=axes, nodelist=nodelist, edgelist=edgelist,
                with_labels=False, edge_color=edgecolor, width=2,
                node_color=node_color, node_size=node_size, alpha=0.7)
        # axes.set_alpha(0.5)
    plt.savefig(save.replace('.png', "_1layer.png"), format="PNG")
    plt.show()
    plt.close()


def write_meta(g, path, dim, n_tree, attr_tree, attr_terminal):
    with open(path, 'w') as f:
        # meta info
        print("[INFO BEGIN]", file=f)
        print(f"grid size: {dim}", file=f)
        print(f"number of trees: {n_tree}", file=f)
        nterminals = []
        for i in range(1, n_tree + 1):
            nodes = [k for (k, v1), (k, v2) in zip(attr_tree.items(), 
                                                    attr_terminal.items()) if v1 == i and v2 == True]
            nterminals.append(len(nodes))
        print(f"number of terminals in each tree: {nterminals}", file=f)
        print("[INFO END]", end='\n\n', file=f)


def write_edges(g, path):
    out = "\n".join([f"{s} - {t}" for s, t in g.edges]) + '\n'
    with open(path, 'a') as f:
        print("[GRAPH EDGES BEGIN]", file=f)
        f.write(out)
        print("[GRAPH EDGES END]", end='\n\n', file=f)
    

def write_terminals(g, path, n_tree, attr_tree, attr_terminal):
    with open(path, 'a') as f:
        print("[TERMINAL NODES BEGIN]", file=f)
        for i in range(1, n_tree + 1):
            print(f"[TERMINAL NODES of TREE {i} BEGIN]", file=f)
            terminals = [f"{k}" for (k, v1), (k, v2) in zip(attr_tree.items(), 
                                                        attr_terminal.items()) if v1 == i and v2 == True]
            out = "\n".join(terminals) + '\n'
            f.write(out)
            print(f"[TERMINAL NODES of TREE {i} END]", file=f)
        print("[TERMINAL NODES END]", file=f)


def write_solution(path, n_tree, attr_tree):
    with open(path, 'w') as f:
        # print path nodes for each tree
        for i in range(1, n_tree + 1):
            print(f"[STEINER TREE {i} BEGIN]", file=f)
            edges = [k for k, v in attr_tree.items() if v == i]
            out = "\n".join([f"{s} - {t}" for s, t in edges]) + "\n"
            f.write(out)
            print(f"[STEINER TREE {i} END]", file=f)
            

def write_constraints(g, path, n_tree, attr_tree):
    if g.graph['constraints']:
        with open(path, 'a') as f:
            print("[CONSTRAINTS BEGIN]", file=f)
            # print path nodes for each tree
            for i in range(1, n_tree + 1):
                print(f"[STEINER TREE {i} CONSTRAINTS BEGIN]", file=f)
                try:
                    constr = [f"{k}: {v}" for k, v in g.graph['constraints'][i].items()]
                except KeyError:
                    pass
                out = "\n".join(constr) + "\n"
                f.write(out)
                print(f"[STEINER TREE {i} CONSTRAINTS END]", file=f)
            print("[CONSTRAINTS END]", file=f)
