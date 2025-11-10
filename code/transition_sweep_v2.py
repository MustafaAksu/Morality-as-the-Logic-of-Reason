#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase–transition sweep for curvature R(t) and relational entropy in a Kuramoto + multi‑cue coupling model.

What’s new vs your version:
- Correct dE/dt using E_e time series (no mix-up with R).
- Returns/uses previous final state for down-sweep (true hysteresis).
- Computes both normalized S̄^R and a monotone Disorder index D = 1 - mean(r_ij).
- Saves CSV + 6 publication-grade PDFs with 95% CI.

Requirements: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# ----------------------- Parameters -----------------------
N_values        = [10, 20, 40]               # system sizes
t_steps         = 100                        # time steps per run
dt              = 0.1
kappa           = 1.0                        # curvature scale
gamma           = 1.0                        # phase-mismatch scale
k_B             = 1.0
T_cog           = 1.0
J_max           = 1.0
forgiveness     = 0.5
seeds           = list(range(42, 52))        # 10 seeds
defect_ratios   = np.arange(0.35, 0.56, 0.02)  # 0.35..0.55 step 0.02
L_beh, L_trust, L_stab = 10, 10, 5           # memory windows
EPS = 1e-12

# ------------------ Multi-cue r_ij function ----------------
def compute_r_ij(t, i, j, coop_history, action_history, types):
    # behavior = i->j recent cooperation mean
    beh = np.mean(coop_history[i, j, -L_beh:]) if t >= L_beh else np.mean(coop_history[i, j, :])
    # trust = j->i recent cooperation mean
    tru = np.mean(coop_history[j, i, -L_trust:]) if t >= L_trust else np.mean(coop_history[j, i, :])
    # semantic proximity (fixed type feature)
    sem = 1 - abs(types[i] - types[j]) / 0.3
    sem = np.clip(sem, 0, 1)
    # stability (low variance = high stability)
    stb = 1 - np.std(action_history[i, j, -L_stab:]) if t >= L_stab else 1 - np.std(action_history[i, j, :])
    stb = np.clip(stb, 0, 1)
    r_ij = 0.4*beh + 0.3*tru + 0.2*sem + 0.1*stb
    return float(np.clip(r_ij, 0.1, 1.0))

# ---------------------- One simulation ---------------------
def run_simulation(N, defect_ratio, seed=None, state=None):
    """
    Runs one trajectory for given N and defector_ratio.
    If state is provided, continues from that state (for hysteresis).
    Returns mean_R, mean_Sbar, mean_D, and final_state.
    """
    if state is None:
        if seed is not None:
            np.random.seed(seed)
        phi = np.random.uniform(0, 2*np.pi, N)
        omega = np.random.uniform(-1, 1, N)
        J = np.random.uniform(0.5, 0.8, (N, N))
        np.fill_diagonal(J, 0.0)
        J = 0.5*(J + J.T)
        types = np.random.uniform(0, 1, N)
        coop_hist  = np.full((N, N, L_beh), 0.5)
        act_hist   = np.full((N, N, L_stab), 0.5)
    else:
        phi, omega, J, types, coop_hist, act_hist = state

    R_t     = np.zeros(t_steps)
    E_e_t   = np.zeros(t_steps)
    Sbar_t  = np.zeros(t_steps)
    D_t     = np.zeros(t_steps)

    for t in range(t_steps):
        # Kuramoto phase update
        dphi = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    dphi[i] += J[i, j] * np.sin(phi[j] - phi[i])
            dphi[i] += omega[i]
        phi += dphi * dt

        # Update couplings via multi-cue resonance proxy
        for i in range(N):
            for j in range(N):
                if i != j:
                    coop_prob = (1.0 - defect_ratio) * forgiveness
                    coop = (np.random.rand() < coop_prob)
                    # update histories (FIFO)
                    coop_hist[i, j, :-1] = coop_hist[i, j, 1:]
                    coop_hist[i, j, -1]  = 1.0 if coop else 0.0
                    act_hist[i, j, :-1]  = act_hist[i, j, 1:]
                    act_hist[i, j, -1]   = coop_prob
                    # r_ij and J update
                    r_ij = compute_r_ij(t, i, j, coop_hist, act_hist, types)
                    J[i, j] = r_ij * J_max
        J = np.clip(J, 0.0, J_max)
        J = 0.5*(J + J.T)

        # Diagnostics
        mask = ~np.eye(N, dtype=bool)
        rvals = (J/J_max)[mask]  # off-diagonal r_ij in [0,1]
        # Ethical energy and its time derivative
        J_avg = np.mean(J[mask])
        E_e   = (k_B*T_cog*np.log((J_max+EPS)/(J_avg+EPS))) if J_avg < J_max else 0.0
        E_e_t[t] = E_e
        dE_dt = (E_e_t[t] - E_e_t[t-1]) / dt if t > 0 else 0.0
        # Curvature: kappa dE/dt + gamma < J_ij sin^2 Δφ >
        sin_sq_sum = 0.0
        count = 0
        for i in range(N):
            for j in range(i+1, N):
                sin_sq_sum += J[i, j] * (np.sin(phi[i]-phi[j])**2)
                count += 1
        sin_sq_avg = sin_sq_sum / max(count, 1)
        R_t[t] = kappa*dE_dt + gamma*sin_sq_avg

        # Relational entropy (normalized) and Disorder index
        S_mean_per_pair = -np.mean(rvals*np.log(rvals+EPS))  # in [0, 1/e]
        Sbar_t[t] = S_mean_per_pair / (1.0/np.e)             # normalized to [0,1]
        D_t[t]    = 1.0 - np.mean(rvals)                     # monotone disorder

    final_state = (phi, omega, J, types, coop_hist, act_hist)
    return float(np.mean(R_t)), float(np.mean(Sbar_t)), float(np.mean(D_t)), final_state

# ----------------------- Sweeps & plots --------------------
R_means, R_cis = {N: [] for N in N_values}, {N: [] for N in N_values}
Sbar_means, Sbar_cis = {N: [] for N in N_values}, {N: [] for N in N_values}
D_means, D_cis = {N: [] for N in N_values}, {N: [] for N in N_values}
dRdp = {N: [] for N in N_values}

# to store results to CSV later
rows = []

for N in N_values:
    # UP sweep (true path-following per seed)
    up_runs_R  = {d: [] for d in defect_ratios}
    up_runs_Sb = {d: [] for d in defect_ratios}
    up_runs_D  = {d: [] for d in defect_ratios}

    # DOWN sweep (continue from last state for hysteresis)
    down_runs_R = {d: [] for d in defect_ratios}

    for seed in seeds:
        state = None
        # up
        for d in defect_ratios:
            Rm, Sm, Dm, state = run_simulation(N, d, seed=seed, state=state)
            up_runs_R[d].append(Rm); up_runs_Sb[d].append(Sm); up_runs_D[d].append(Dm)
        # down (reuse the last state)
        for d in reversed(defect_ratios):
            Rm, _, _, state = run_simulation(N, d, seed=seed, state=state)
            down_runs_R[d].append(Rm)

    # aggregate (means + 95% CI)
    for d in defect_ratios:
        R_mu, R_ci = np.mean(up_runs_R[d]), 1.96*sem(up_runs_R[d])
        S_mu, S_ci = np.mean(up_runs_Sb[d]), 1.96*sem(up_runs_Sb[d])
        D_mu, D_ci = np.mean(up_runs_D[d]), 1.96*sem(up_runs_D[d])
        R_means[N].append(R_mu);  R_cis[N].append(R_ci)
        Sbar_means[N].append(S_mu); Sbar_cis[N].append(S_ci)
        D_means[N].append(D_mu);  D_cis[N].append(D_ci)
        rows.append([N, d, R_mu, R_ci, S_mu, S_ci, D_mu, D_ci])

    # numerical derivative dR/dp
    R_vals = np.array(R_means[N])
    dRdp[N] = np.gradient(R_vals, defect_ratios)

# -------- Save CSV --------
import csv
with open('transition_results.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['N','defector_ratio','R_mean','R_95CI','Sbar_mean','Sbar_95CI','D_mean','D_95CI'])
    w.writerows(rows)

# --------- Plots ----------
plt.figure(figsize=(10,6))
for N in N_values:
    plt.errorbar(defect_ratios, R_means[N], yerr=R_cis[N], label=f'N={N}', capsize=3)
plt.xlabel('Defector ratio')
plt.ylabel('Mean curvature R(t)')
plt.title('R(t) vs Defector ratio (with 95% CI)')
plt.legend(); plt.tight_layout(); plt.savefig('phase_transition_R.pdf')

plt.figure(figsize=(10,6))
for N in N_values:
    plt.errorbar(defect_ratios, Sbar_means[N], yerr=Sbar_cis[N], label=f'N={N}', capsize=3)
plt.xlabel('Defector ratio')
plt.ylabel('Mean normalized relational entropy $\\bar{S}^{\\mathsf{R}}$')
plt.title('$\\bar{S}^{\\mathsf{R}}$ vs Defector ratio (with 95% CI)')
plt.legend(); plt.tight_layout(); plt.savefig('phase_transition_Sbar.pdf')

plt.figure(figsize=(10,6))
for N in N_values:
    plt.errorbar(defect_ratios, D_means[N], yerr=D_cis[N], label=f'N={N}', capsize=3)
plt.xlabel('Defector ratio')
plt.ylabel('Disorder index $D=1-\\langle r_{ij}\\rangle$')
plt.title('Monotone disorder vs Defector ratio (with 95% CI)')
plt.legend(); plt.tight_layout(); plt.savefig('phase_transition_disorder.pdf')

plt.figure(figsize=(10,6))
for N in N_values:
    plt.plot(defect_ratios, dRdp[N], label=f'N={N}')
plt.xlabel('Defector ratio')
plt.ylabel('dR/dp')
plt.title('Critical point: peak of dR/dp')
plt.legend(); plt.tight_layout(); plt.savefig('dRdp_peak.pdf')

# Simple hysteresis visual (up vs down) for N=20
plt.figure(figsize=(10,6))
Nref = 20
# reconstruct down means from last loop (same averaging as up)
# (for display, we take means over seeds we collected into down_runs_R during last N loop,
#  but since we reused local 'down_runs_R' per N in the loop above, we quickly recompute here)
# quick recompute down means for Nref:
# (a compact recompute; not the most efficient but keeps code readable)
down_means = []
up_means   = R_means[Nref]
# recompute down means:
down_runs_R = {d: [] for d in defect_ratios}
for seed in seeds:
    state = None
    for d in defect_ratios:
        _, _, _, state = run_simulation(Nref, d, seed=seed, state=state)
    for d in reversed(defect_ratios):
        Rm, _, _, state = run_simulation(Nref, d, seed=seed, state=state)
        down_runs_R[d].append(Rm)
down_means = [np.mean(down_runs_R[d]) for d in defect_ratios[::-1]]

plt.plot(defect_ratios, up_means, label=f'Up sweep (N={Nref})', linestyle='-')
plt.plot(defect_ratios[::-1], down_means, label=f'Down sweep (N={Nref})', linestyle='--')
plt.xlabel('Defector ratio'); plt.ylabel('Mean R(t)')
plt.title('Hysteresis test (path dependence) for N=20')
plt.legend(); plt.tight_layout(); plt.savefig('hysteresis_R.pdf')

plt.figure(figsize=(10,6))
for N in N_values:
    plt.plot(defect_ratios, R_means[N], label=f'N={N}')
plt.xlabel('Defector ratio')
plt.ylabel('Mean R(t)')
plt.title('Finite-size effect: R(t) vs defector ratio')
plt.legend(); plt.tight_layout(); plt.savefig('finite_size_transition.pdf')
