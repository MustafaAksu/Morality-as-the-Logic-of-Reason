# Script Name: ipd_resonance_simulation_v5.py
# Purpose: Validate "Morality as the Logic of Reason" with a 10-agent IPD.
#          Fixes the "missing blue line" by storing separate S_R_bar histories
#          for each (defector_rate, delta) pair and plotting the correct series.
#          Generates: ipd_chart.pdf (coop vs δ), entropy_chart.pdf (S̄^R over time),
#          rij_heatmap_*.pdf (final r_ij heatmaps), and simulation_results.csv.
#
# Authors: Mustafa Aksu, with AI collaborators (Grok by xAI, ChatGPT by OpenAI)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------- Parameters (literature-grounded) -------------
n_agents = 10            # Axelrod 1984
n_rounds = 200           # sufficient for convergence
noise = 0.05             # Nowak 2006
delta_values = [0.5, 0.75, 0.95]  # Frederick 2002
k = 0.1                  # impatience (Frederick 2002)
n = 2                    # quadratic-like discount (Ainslie 1975)
forgive_base = 0.1       # base GTFT forgiveness (Santos et al. 2018)
defector_rates = [0.2, 0.5]  # 20% and 50%
w_k = [0.4, 0.3, 0.2, 0.1]   # behavior, semantic, trust, stability

# Payoff matrix for IPD (C=1, D=0)
payoffs = {(1, 1): (3, 3), (1, 0): (0, 5), (0, 1): (5, 0), (0, 0): (1, 1)}

# Reproducibility & directories
np.random.seed(42)
paper_dir = os.path.join("..", "paper")
os.makedirs(paper_dir, exist_ok=True)

# Max entropy for normalization: sum over N(N-1) off-diagonals, max at r=1/e
S_R_max = n_agents * (n_agents - 1) / np.e

# ------------- Agent types (semantic "prior") -------------
# Cooperative bias in [0.7, 1.0]
agent_types = np.random.uniform(0.7, 1.0, n_agents)

# ------------- Strategy helpers -------------
def play_gtft(history, i, j, forgive_rate, noise):
    """Adaptive Generous-TFT with noise and trust-based forgiveness."""
    if np.random.random() < noise:
        return np.random.choice([0, 1])
    if len(history) == 0:
        return 1  # cooperate first
    last_n = min(len(history), 10)
    opp_actions = [history[t][j][i] for t in range(-last_n, 0)] if last_n > 0 else []
    trust = np.mean(opp_actions) if opp_actions else 1.0
    adaptive_forgive = forgive_rate * (1 - trust)  # forgive more when trust is low
    if np.random.random() < adaptive_forgive:
        return 1
    return history[-1][j][i] if history else 1

def compute_E_c(opp_prob, k=0.1, t=1, n=2):
    """
    Ethical energy for next-round decision.
    opp_prob = P(opponent cooperates)
    """
    # H-A for our action given opponent action
    # If we cooperate: (3 if opp coop, 0 if opp defect)
    # If we defect:    (5 if opp coop, 1 if opp defect)
    E_c_coop = opp_prob * (3 - 0) / (1 + k * t) ** n + (1 - opp_prob) * (0 - 0) / (1 + k * t) ** n
    E_c_defect = opp_prob * (5 - 0) / (1 + k * t) ** n + (1 - opp_prob) * (1 - 0) / (1 + k * t) ** n
    return E_c_coop, E_c_defect

# ------------- Storage -------------
results = []  # list of dicts (per (defector_rate, delta))
S_R_bar_histories = {}  # (defector_rate, delta) -> list of S̄^R over rounds

# ------------- Simulation loop -------------
for defector_rate in defector_rates:
    for delta in delta_values:
        print(f"\n--- Simulation with {defector_rate*100:.0f}% Defectors, δ={delta} ---")

        # Initialize defectors
        defectors = np.random.choice(n_agents, int(n_agents * defector_rate), replace=False)

        # Initial resonance r_ij
        r_ij = np.random.uniform(0.5, 1.0, (n_agents, n_agents))
        np.fill_diagonal(r_ij, 1.0)

        scores = np.zeros(n_agents)
        cooperation_rates = np.zeros(n_agents)
        history = []
        S_R_history = []
        S_R_bar_history = []

        for rnd in range(n_rounds):
            round_actions = np.ones((n_agents, n_agents), dtype=int)

            # Choose actions
            for i in range(n_agents):
                for j in range(n_agents):
                    if i == j:
                        continue
                    if i in defectors and np.random.random() < 0.8:
                        round_actions[i, j] = 0  # probabilistic defection
                    else:
                        # Estimate opponent coop prob from recent history
                        last_n = min(len(history), 10)
                        opp_actions = [history[t][j][i] for t in range(-last_n, 0)] if last_n > 0 else []
                        opp_prob = np.mean(opp_actions) if opp_actions else 1.0
                        E_c_coop, E_c_defect = compute_E_c(opp_prob, k=k, t=1, n=n)
                        round_actions[i, j] = 1 if E_c_coop >= E_c_defect else play_gtft(history, i, j, forgive_base, noise)

            history.append(round_actions)

            # Update r_ij with multi-cue blend
            for i in range(n_agents):
                for j in range(n_agents):
                    if i == j:
                        continue
                    # Behavior and trust from last 10 rounds
                    last_n = min(len(history), 10)
                    beh_series = [history[t][i][j] for t in range(-last_n, 0)] if last_n > 0 else [1]
                    tr_series  = [history[t][j][i] for t in range(-last_n, 0)] if last_n > 0 else [1]
                    behavior = float(np.mean(beh_series))
                    trust = float(np.mean(tr_series))
                    # Semantic proximity (bounded to [0,1])
                    semantic = 1.0 - abs(agent_types[i] - agent_types[j]) / 0.3
                    semantic = float(np.clip(semantic, 0.0, 1.0))
                    # Stability: 1 - std of our actions to j over last 5 rounds
                    last_m = min(len(history), 5)
                    stab_series = [history[t][i][j] for t in range(-last_m, 0)] if last_m > 0 else [1]
                    stability = 1.0 - float(np.std(stab_series))
                    stability = float(np.clip(stability, 0.0, 1.0))

                    r_ij[i, j] = (
                        w_k[0] * behavior +
                        w_k[1] * semantic +
                        w_k[2] * trust +
                        w_k[3] * stability
                    )

            # Compute S^R and normalized S̄^R for this round
            S_R = -np.sum(r_ij * np.log(r_ij + 1e-12))
            S_R_bar = S_R / S_R_max
            S_R_history.append(S_R)
            S_R_bar_history.append(S_R_bar)

            # Update discounted scores and accumulate cooperation rates
            for i in range(n_agents):
                for j in range(n_agents):
                    if i == j:
                        continue
                    a_i, a_j = int(round_actions[i, j]), int(round_actions[j, i])
                    scores[i] += payoffs[(a_i, a_j)][0] * (delta ** rnd)
            cooperation_rates += np.mean(round_actions, axis=1)  # per agent mean over partners

        # Store per-curve time series (this is the key fix)
        S_R_bar_histories[(defector_rate, delta)] = S_R_bar_history.copy()

        # Save final r_ij for heatmaps
        np.save(f'final_rij_{int(defector_rate*100)}pct_delta_{delta}.npy', r_ij)

        # Final aggregates (per (defector_rate, delta))
        cooperation_rates /= n_rounds
        avg_scores = scores / n_rounds
        avg_S_R = float(np.mean(S_R_history))
        avg_S_R_bar = float(np.mean(S_R_bar_history))

        results.append({
            'defector_rate': defector_rate,
            'delta': delta,
            'avg_scores': avg_scores.tolist(),
            'avg_coop_rates': cooperation_rates.tolist(),
            'avg_S_R': avg_S_R,
            'avg_S_R_bar': avg_S_R_bar
        })

        print(f"Average Scores: {avg_scores.round(3)}")
        print(f"Cooperation Rates: {cooperation_rates.round(4)}")
        print(f"Average Relational Entropy (S^R): {avg_S_R:.3f}")
        print(f"Normalized Relational Entropy (S^R_bar): {avg_S_R_bar:.3f}")

# Save summary CSV
pd.DataFrame(results).to_csv('simulation_results.csv', index=False)
print("Results saved to simulation_results.csv")

# ------------- Plot 1: cooperation vs δ -------------
fig, ax = plt.subplots(figsize=(6.2, 4.0))
palette = {0.2: '#1f77b4', 0.5: '#ff7f0e'}  # blue for 20%, orange for 50%
for dr in defector_rates:
    # mean of per-agent coop rates per δ
    coop_means = []
    deltas = []
    for r in results:
        if r['defector_rate'] == dr:
            coop_means.append(float(np.mean(r['avg_coop_rates'])))
            deltas.append(r['delta'])
    # ensure consistent sort by δ
    deltas, coop_means = zip(*sorted(zip(deltas, coop_means)))
    ax.plot(deltas, coop_means, marker='o', lw=2, color=palette[dr],
            label=f'{int(dr*100)}% Defectors')
ax.set_xlabel('Discount Factor ($\\delta$)')
ax.set_ylabel('Mean Cooperation Rate')
ax.set_title('IPD: Cooperation Rate vs Discount Factor')
ax.set_ylim(0, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(paper_dir, 'ipd_chart.pdf'))
print('Saved ../paper/ipd_chart.pdf')

# ------------- Plot 2: S̄^R over rounds for δ=0.95 -------------
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for dr in defector_rates:
    key = (dr, 0.95)
    if key in S_R_bar_histories:
        series = S_R_bar_histories[key]
        ax.plot(range(len(series)), series, lw=2, color=palette[dr],
                label=f'{int(dr*100)}% Defectors')
    else:
        print(f"[WARN] No S_R_bar history for {int(dr*100)}% at δ=0.95")
ax.set_xlabel('Round')
ax.set_ylabel('Normalized Relational Entropy ($\\bar{S}^{\\mathsf{R}}$)')
ax.set_title('Normalized Relational Entropy Over 200 Rounds ($\\delta=0.95$)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(paper_dir, 'entropy_chart.pdf'))
print('Saved ../paper/entropy_chart.pdf')

# ------------- Plot 3: heatmaps for each (dr, δ) -------------
for dr in defector_rates:
    for delta in delta_values:
        fname = f'final_rij_{int(dr*100)}pct_delta_{delta}.npy'
        if not os.path.exists(fname):
            print(f"[WARN] Missing {fname}, skipping heatmap.")
            continue
        R = np.load(fname)
        plt.figure(figsize=(4.8, 4.0))
        sns.heatmap(R, vmin=0, vmax=1, cmap='viridis', square=True,
                    cbar_kws={'label': '$r_{ij}$'})
        plt.title(f'Final $r_{{ij}}$ Matrix ({int(dr*100)}% Defectors, $\\delta={delta}$)')
        plt.tight_layout()
        out = os.path.join(paper_dir, f'rij_heatmap_{int(dr*100)}pct_delta_{delta}.pdf')
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")
