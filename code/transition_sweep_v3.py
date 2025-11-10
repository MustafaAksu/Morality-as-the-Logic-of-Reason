#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transition_sweep_v3.py — Track A: Curvature ⇄ Disorder

- 10–40 agent IPD/Kuramoto hybrid with multi-cue r_ij
- Sweeps defector ratio and δ
- Logs normalized relational entropy (S_R_bar), moral curvature R(t)
- NEW: logs Forman-style mean edge curvature (kappa_bar) per run
- Produces paper/curvature_vs_entropy.pdf
- Writes code/simulation_results.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------- Parameters ---------------------
N_LIST        = [10, 20, 40]      # agent counts
N_ROUNDS      = 200               # time steps per run
DT            = 0.1               # Kuramoto step
NOISE         = 0.05              # action noise
DELTA_LIST    = [0.5, 0.75, 0.95] # discount factors
K_IMPATIENCE  = 0.1               # k in E_c
N_SHAPE       = 2                 # n in E_c
FORGIVENESS   = 0.5               # base GTFT forgiveness
DEF_RATES     = [0.2, 0.5]        # defector fractions
SEEDS         = [42, 43, 44]      # reproducibility

# Paths
ROOT = Path(".")
PAPER_DIR = ROOT / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = ROOT / "code" / "simulation_results.csv"
Path("code").mkdir(exist_ok=True)

# Max entropy factor for normalization
def S_R_max(n): 
    return (n * (n - 1)) / np.e

# --------------------- Curvature (Forman) ---------------------
def _neighbors(W, i, exclude=None):
    idx = np.where(W[i] > 0)[0]
    if exclude is not None:
        idx = idx[idx != exclude]
    return idx

def mean_forman_curvature(W: np.ndarray) -> float:
    """Mean Forman curvature for symmetric nonnegative W (zero diagonal)."""
    n = W.shape[0]
    deg = W.sum(axis=1) + 1e-12
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            w_ij = W[i, j]
            if w_ij <= 0:
                continue
            Ni = _neighbors(W, i, j)
            Nj = _neighbors(W, j, i)
            core = w_ij * (1.0/deg[i] + 1.0/deg[j])
            pen_i = np.sum(W[i, Ni] / np.sqrt((w_ij + 1e-12) * (W[i, Ni] + 1e-12)))
            pen_j = np.sum(W[j, Nj] / np.sqrt((w_ij + 1e-12) * (W[j, Nj] + 1e-12)))
            vals.append(float(core - (pen_i + pen_j)))
    return float(np.mean(vals)) if vals else 0.0

# --------------------- Simulation core ---------------------
def play_gtft(history, i, j):
    """Adaptive Generous-TFT with noise and trust-based forgiveness."""
    if np.random.random() < NOISE:
        return np.random.choice([0, 1])
    if len(history) == 0:
        return 1
    last_n = min(len(history), 10)
    opp_actions = [history[t][j][i] for t in range(-last_n, 0)] if last_n > 0 else []
    trust = float(np.mean(opp_actions)) if opp_actions else 1.0
    adaptive_forgive = FORGIVENESS * (1 - trust)
    return 1 if np.random.random() < max(adaptive_forgive, 0.0) else int(opp_actions[-1] if opp_actions else 1)

def compute_Ec(opp_prob, k=0.1, t=1.0, n=2):
    disc = (1 + k * t) ** n
    EcC = (opp_prob * 3.0 + (1 - opp_prob) * 0.0) / disc
    EcD = (opp_prob * 5.0 + (1 - opp_prob) * 1.0) / disc
    return EcC, EcD

def run_scenario(n_agents, n_rounds, delta, def_rate, seed):
    np.random.seed(seed)
    r_ij = np.random.uniform(0.5, 1.0, (n_agents, n_agents))
    np.fill_diagonal(r_ij, 0.0)
    scores = np.zeros(n_agents)
    coop_rates = np.zeros(n_agents)
    history = []
    S_R_hist, S_R_bar_hist, R_hist, kappa_hist = [], [], [], []
    types = np.random.uniform(0.7, 1.0, n_agents)
    defectors = np.random.choice(n_agents, int(n_agents * def_rate), replace=False)

    def update_multi_cue_rij():
        nonlocal r_ij
        newR = np.zeros_like(r_ij)
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                last_n = min(len(history), 10)
                beh = np.mean([history[t][i][j] for t in range(-last_n, 0)]) if last_n else 1.0
                tru = np.mean([history[t][j][i] for t in range(-last_n, 0)]) if last_n else 1.0
                last_m = min(len(history), 5)
                stab = 1.0 - (np.std([history[t][i][j] for t in range(-last_m, 0)]) if last_m else 0.0)
                sem  = 1.0 - abs(types[i] - types[j]) / 0.3
                sem  = np.clip(sem, 0.0, 1.0)
                val = 0.4*beh + 0.3*tru + 0.2*sem + 0.1*stab
                newR[i, j] = np.clip(val, 0.0, 1.0)
        np.fill_diagonal(newR, 0.0)
        r_ij[...] = newR

    phi = np.random.uniform(0, 2*np.pi, n_agents)

    for t in range(n_rounds):
        actions = np.ones((n_agents, n_agents), dtype=int)
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                if i in defectors and np.random.random() < 0.8:
                    actions[i, j] = 0
                else:
                    last_n = min(len(history), 10)
                    opp_actions = [history[k][j][i] for k in range(-last_n, 0)] if last_n else []
                    opp_prob = np.mean(opp_actions) if opp_actions else 1.0
                    EcC, EcD = compute_Ec(opp_prob, k=K_IMPATIENCE, t=1.0, n=N_SHAPE)
                    actions[i, j] = 1 if EcC >= EcD else play_gtft(history, i, j)

        history.append(actions.copy())
        coop_rates += actions.mean(axis=1)
        update_multi_cue_rij()

        W = r_ij.copy()
        np.fill_diagonal(W, 0.0)
        S_R = -np.sum(W * np.log(W + 1e-12))
        S_R_bar = S_R / S_R_max(n_agents)
        S_R_hist.append(S_R)
        S_R_bar_hist.append(S_R_bar)
        kappa_hist.append(mean_forman_curvature(W))

        phi += DT * np.sum(W * np.sin(phi - phi.reshape(-1,1)), axis=1)
        angles = phi.reshape(-1,1) - phi.reshape(1,-1)
        mask = ~np.eye(n_agents, dtype=bool)
        sin2 = np.sin(angles[mask])**2
        weights = W[mask]
        R_t = float((weights * sin2).mean()) / ((1 + K_IMPATIENCE*1.0) ** N_SHAPE)
        R_hist.append(R_t)

    coop_avg = float((coop_rates / n_rounds).mean())
    return {
        "mean_S_R": float(np.mean(S_R_hist)),
        "mean_S_R_bar": float(np.mean(S_R_bar_hist)),
        "mean_R": float(np.mean(R_hist)),
        "mean_kappa": float(np.mean(kappa_hist)),
        "coop_avg": coop_avg
    }

# --------------------- Main ---------------------
def main():
    rows = []
    for n in [20]:
        for dr in DEF_RATES:
            for delta in DELTA_LIST:
                for seed in SEEDS:
                    res = run_scenario(n, N_ROUNDS, delta, dr, seed)
                    rows.append({
                        "N": n, "defector_rate": dr, "delta": delta,
                        "mean_S_R": res["mean_S_R"],
                        "mean_S_R_bar": res["mean_S_R_bar"],
                        "mean_R": res["mean_R"],
                        "mean_kappa": res["mean_kappa"],
                        "coop_mean": res["coop_avg"]
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # Curvature vs Entropy
    plt.figure(figsize=(6.2,4.0))
    sns.scatterplot(data=df, x="mean_S_R_bar", y="mean_kappa", hue="defector_rate", style="delta", s=60)
    r = np.corrcoef(df["mean_S_R_bar"], df["mean_kappa"])[0,1]
    z = np.polyfit(df["mean_S_R_bar"], df["mean_kappa"], 1)
    xs = np.linspace(df["mean_S_R_bar"].min(), df["mean_S_R_bar"].max(), 100)
    plt.plot(xs, z[0]*xs + z[1], color='k', alpha=0.6, label="OLS trend")
    plt.title(f"Curvature vs Entropy (r={r:.2f})")
    plt.xlabel("Mean $\\bar S^{\\mathsf{R}}$")
    plt.ylabel("Mean Forman curvature $\\overline{\\kappa}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PAPER_DIR / "curvature_vs_entropy.pdf")
    plt.close()

    print(f"✅ Wrote results to {OUT_CSV} and paper/curvature_vs_entropy.pdf")

if __name__ == "__main__":
    main()
