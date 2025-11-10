# analysis/curvature.py
import numpy as np

def _neighbors(W, i, exclude=None):
    nbh = np.where(W[i] > 0)[0]
    if exclude is not None:
        nbh = nbh[nbh != exclude]
    return nbh

def scalar_forman_curvature(W: np.ndarray) -> float:
    """
    Simplified weighted Forman curvature averaged over edges.
    W: symmetric NxN, W[i,i]=0, W[i,j]>=0 interpreted as r_ij (trust).
    Returns mean edge curvature (scalar).
    """
    n = W.shape[0]
    # degree-like weight (row-sum) with small epsilon for stability
    deg = W.sum(axis=1) + 1e-12
    F_vals = []
    for i in range(n):
        for j in range(i+1, n):
            w_ij = W[i, j]
            if w_ij <= 0: 
                continue
            # neighbor contributions (excluding the counterpart)
            Ni = _neighbors(W, i, exclude=j)
            Nj = _neighbors(W, j, exclude=i)
            # edge-local Forman: positive core minus neighbor penalties
            core = w_ij * (1.0/deg[i] + 1.0/deg[j])
            pen_i = np.sum(W[i, Ni] / np.sqrt((w_ij+1e-12) * (W[i, Ni]+1e-12)))
            pen_j = np.sum(W[j, Nj] / np.sqrt((w_ij+1e-12) * (W[j, Nj]+1e-12)))
            F_ij = core - (pen_i + pen_j)
            F_vals.append(F_ij)
    return float(np.mean(F_vals)) if F_vals else 0.0
