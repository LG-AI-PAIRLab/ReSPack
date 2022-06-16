def get_bbox(netlists):
    """
    calculate width^2 + height^2 of bounding box

    Returns
    ----------
    bboxs : dict. { tuple of net_id : scalar (width^2 + height^2)}
    """
    bboxs = {}
    for key, net in netlists.items():
        st, left = net[0], net[1:]
        min_, max_ = st, st
        for coord in left:
            if min_ > coord:
                min_ = coord
            elif max_ < coord:
                max_ = coord
        # distance^2
        bboxs[key] = (max_[0] - min_[0]) ** 2 + (max_[1] - min_[1]) ** 2
    return bboxs


def init_startnodes(terminals):
    """
    select start nodes among terminal points
    """
    import random

    start = random.choice(terminals)
    terminals.remove(start)
    return start, terminals


def init_visited(nodes, starts):
    max_ = float('inf') 
    visited = {n: max_ for n in nodes}
    for st in starts:
        visited[st] = 0
    return visited


def backtrace(history):
    visited_edges = set()
    st = history[0]
    for ed in history[1:]:
        edge = (st, ed) if st < ed else (ed, st)
        visited_edges.add(edge)
        st = ed
    return visited_edges
