# analysis/ollivier.py (optional)
import numpy as np
from itertools import product

def one_hop_measure(W, i):
    nbrs = np.where(W[i] > 0)[0]
    if nbrs.size == 0:
        return {i:1.0}
    w = W[i, nbrs]
    p = w / w.sum()
    return dict(zip(nbrs.tolist(), p.tolist()))

def wasserstein1(m_i, m_j, ground_dist):
    # tiny Hungarian-like O(n^2) matcher for one-hop support
    # m_i, m_j: dict node->mass, ground_dist(u,v): distance
    nodes_i = list(m_i.keys()); nodes_j = list(m_j.keys())
    P = np.zeros((len(nodes_i), len(nodes_j)))
    for a,u in enumerate(nodes_i):
        for b,v in enumerate(nodes_j):
            P[a,b] = ground_dist(u,v)
    # simple greedy for illustration; for rigor use linear programming
    supply = np.array([m_i[u] for u in nodes_i], float)
    demand = np.array([m_j[v] for v in nodes_j], float)
    cost = 0.0
    # greedy: move from closest pairs progressively
    pairs = [(P[a,b], a,b) for a,b in product(range(len(nodes_i)), range(len(nodes_j)))]
    for _, a,b in sorted(pairs):
        flow = min(supply[a], demand[b])
        if flow>0:
            cost += flow * P[a,b]
            supply[a] -= flow; demand[b] -= flow
    return cost

def mean_ollivier_curvature(W):
    n = W.shape[0]
    def dist(u,v): return 0.0 if u==v else 1.0 if W[u,v]>0 else 2.0
    kappa_vals = []
    for i in range(n):
        for j in range(i+1,n):
            if W[i,j]<=0: continue
            m_i = one_hop_measure(W,i)
            m_j = one_hop_measure(W,j)
            W1 = wasserstein1(m_i,m_j, dist)
            kappa = 1.0 - W1/1.0  # edge distance=1
            kappa_vals.append(kappa)
    return float(np.mean(kappa_vals)) if kappa_vals else 0.0
