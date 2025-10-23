# Script Name: ipd_resonance_simulation.py
# Purpose: Validates "Morality as the Logic of Reason" via a 10-agent IPD.
#          - Adaptive Generous-TFT (GTFT) under defector rates (20%, 50%)
#          - Discount factors δ ∈ {0.5, 0.75, 0.95}
#          - Multi-cue resonance r_ij with weights [0.4, 0.3, 0.2, 0.1]
#          - Ethical energy E_c = (H-A)/(1 + k t)^n drives decisions
#          - Tracks S^R and normalized S^R (S^R_max = N(N-1)/e)
#          - Saves charts: ipd_chart.pdf, entropy_chart.pdf, heatmaps for each δ
#
# Authors: Mustafa Aksu, with AI collaborators (Grok by xAI, ChatGPT by OpenAI)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# Parameters
# --------------------
n_agents = 10
n_rounds = 200
noise = 0.05
delta_values = [0.5, 0.75, 0.95]
k = 0.1    # impatience
n = 2      # ~quadratic discounting
forgive_base = 0.1
defector_rates = [0.2, 0.5]  # 20% and 50%
w_k = [0.4, 0.3, 0.2, 0.1]   # behavior, semantic, trust, stability
S_R_max = n_agents * (n_agents - 1) / np.e

# File/dir setup
ROOT = Path(__file__).resolve().parents[1]  # repo root
PAPER_DIR = ROOT / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

# IPD payoffs (C=1, D=0)
payoffs = {(1, 1): (3, 3), (1, 0): (0, 5), (0, 1): (5, 0), (0, 0): (1, 1)}

# Reproducibility
np.random.seed(42)
agent_types = np.random.uniform(0.7, 1.0, n_agents)  # semantic prior

def format_delta_for_name(delta: float) -> str:
    """Return a LaTeX-safe delta string: 0.5 -> 0p50, 0.95 -> 0p95."""
    return str(delta).replace(".", "p")

def play_gtft(history, i, j):
    """Adaptive GTFT with simple forgiveness & noise."""
    if np.random.random() < noise:
        return np.random.choice([0, 1])
    if len(history) == 0:
        return 1  # Cooperate first
    last_n = min(len(history), 10)
    opp_actions = [history[t][j][i] for t in range(-last_n, 0)] if last_n > 0 else []
    trust = np.mean(opp_actions) if opp_actions else 1.0
    adaptive_forgive = forgive_base * (1 - trust)
    if np.random.random() < adaptive_forgive:
        return 1
    return history[-1][j][i]  # mirror

def compute_E_c(opp_prob, k=0.1, t=1, n=2):
    """Ethical energy for cooperate vs defect against prob(opp cooperates)=opp_prob."""
    H_A_coop = [(payoffs[(1, a)][0], 0) for a in [1, 0]]   # (h,a) for opp C/D
    H_A_def  = [(payoffs[(0, a)][0], 0) for a in [1, 0]]   # (h,a) for opp C/D
    disc = (1 + k * t) ** n
    E_c_coop = opp_prob * (H_A_coop[0][0]) / disc + (1 - opp_prob) * (H_A_coop[1][0]) / disc
    E_c_def  = opp_prob * (H_A_def[0][0])  / disc + (1 - opp_prob) * (H_A_def[1][0])  / disc
    return E_c_coop, E_c_def

results = []  # one dict per (defector_rate, delta)

for defector_rate in defector_rates:
    for delta in delta_values:
        print(f"\n--- Simulation with {defector_rate*100:.0f}% Defectors, δ={delta} ---")

        defectors = np.random.choice(n_agents, int(n_agents * defector_rate), replace=False)
        r_ij = np.random.uniform(0.5, 1.0, (n_agents, n_agents))
        np.fill_diagonal(r_ij, 1.0)
        scores = np.zeros(n_agents)
        cooperation_rates = np.zeros(n_agents)
        history = []
        S_R_history = []
        S_R_bar_history = []

        for t in range(n_rounds):
            round_actions = np.ones((n_agents, n_agents))
            # Decide actions
            for i in range(n_agents):
                for j in range(n_agents):
                    if i == j:
                        continue
                    if i in defectors and np.random.random() < 0.8:
                        round_actions[i, j] = 0
                    else:
                        last_n = min(len(history), 10)
                        opp_actions = [history[τ][j][i] for τ in range(-last_n, 0)] if last_n > 0 else []
                        opp_prob = np.mean(opp_actions) if opp_actions else 1.0
                        EcC, EcD = compute_E_c(opp_prob, k=k, t=1, n=n)
                        round_actions[i, j] = 1 if EcC >= EcD else play_gtft(history, i, j)

            history.append(round_actions)

            # Update r_ij with multi-cue blend
            for i in range(n_agents):
                for j in range(n_agents):
                    if i == j:
                        continue
                    # recent behavior consistency & trust
                    last10 = min(len(history), 10)
                    last5  = min(len(history), 5)
                    beh = np.mean([history[τ][i][j] for τ in range(-last10, 0)]) if last10 else 1.0
                    tru = np.mean([history[τ][j][i] for τ in range(-last10, 0)]) if last10 else 1.0
                    stab = 1 - np.std([history[τ][i][j] for τ in range(-last5, 0)]) if last5 else 1.0
                    sem = 1 - abs(agent_types[i] - agent_types[j]) / 0.3
                    sem = np.clip(sem, 0.0, 1.0)
                    r_ij[i, j] = w_k[0]*beh + w_k[1]*sem + w_k[2]*tru + w_k[3]*stab
                    r_ij[i, j] = np.clip(r_ij[i, j], 1e-6, 1.0)

            # Entropy (unnormalized & normalized)
            S_R = -np.sum(r_ij * np.log(r_ij))
            S_R_bar = S_R / S_R_max
            S_R_history.append(float(S_R))
            S_R_bar_history.append(float(S_R_bar))

            # Discounted scoring & cooperation accounting
            for i in range(n_agents):
                for j in range(n_agents):
                    if i == j:
                        continue
                    aij = int(round_actions[i, j])
                    aji = int(round_actions[j, i])
                    scores[i] += payoffs[(aij, aji)][0] * (delta ** t)

            cooperation_rates += np.mean(round_actions, axis=1)

        # Save per-scenario r_ij for heatmaps (LaTeX-safe & decimal)
        dstr = format_delta_for_name(delta)  # 0.5 -> 0p5
        np.save(ROOT / f"code/final_rij_{int(defector_rate*100)}pct_delta_{dstr}.npy", r_ij)
        np.save(ROOT / f"code/final_rij_{int(defector_rate*100)}pct_delta_{delta}.npy", r_ij)

        # Aggregate
        cooperation_rates /= n_rounds
        avg_scores = scores / n_rounds
        result = {
            "defector_rate": defector_rate,
            "delta": delta,
            "avg_scores": avg_scores.tolist(),
            "avg_coop_rates": cooperation_rates.tolist(),
            "avg_S_R": float(np.mean(S_R_history)),
            "avg_S_R_bar": float(np.mean(S_R_bar_history)),
            # store the full series so the plot uses the right curve
            "S_R_bar_series": S_R_bar_history,
        }
        results.append(result)

        print(f"Average Scores: {avg_scores}")
        print(f"Cooperation Rates: {cooperation_rates}")
        print(f"Average Relational Entropy (S^R): {result['avg_S_R']:.3f}")
        print(f"Normalized Relational Entropy (S^R_bar): {result['avg_S_R_bar']:.3f}")

# Save table for transparency
pd.DataFrame(results).to_csv(ROOT / "code/simulation_results.csv", index=False)
print("Results saved to code/simulation_results.csv")

# --------------------
# Plots
# --------------------
# 1) Cooperation vs δ
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for dr in defector_rates:
    rlist = [r for r in results if r["defector_rate"] == dr]
    rlist = sorted(rlist, key=lambda x: x["delta"])
    deltas = [r["delta"] for r in rlist]
    coop_means = [float(np.mean(r["avg_coop_rates"])) for r in rlist]
    ax.plot(deltas, coop_means, marker='o', label=f"{int(dr*100)}% Defectors")
ax.set_xlabel("Discount Factor ($\\delta$)")
ax.set_ylabel("Mean Cooperation Rate")
ax.set_title("IPD: Cooperation Rate vs Discount Factor")
ax.legend()
plt.tight_layout()
plt.savefig(PAPER_DIR / "ipd_chart.pdf")
print(f"Saved {PAPER_DIR / 'ipd_chart.pdf'}")

# 2) Normalized S^R over rounds at δ=0.95 (two distinct series!)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for dr in defector_rates:
    r95 = [r for r in results if r["defector_rate"] == dr and abs(r["delta"] - 0.95) < 1e-9]
    if not r95:
        continue
    sr = r95[0]["S_R_bar_series"]
    ax.plot(range(len(sr)), sr, label=f"{int(dr*100)}% Defectors")
ax.set_xlabel("Round")
ax.set_ylabel("Normalized Relational Entropy ($\\bar{S}^{\\mathsf{R}}$)")
ax.set_title("Normalized Relational Entropy Over 200 Rounds ($\\delta=0.95$)")
ax.legend()
plt.tight_layout()
plt.savefig(PAPER_DIR / "entropy_chart.pdf")
print(f"Saved {PAPER_DIR / 'entropy_chart.pdf'}")

# 3) Heatmaps for every δ, both naming styles (safe/decimal)
for dr in defector_rates:
    for delta in delta_values:
        dstr = format_delta_for_name(delta)
        # load from the LaTeX-safe name we saved
        R = np.load(ROOT / f"code/final_rij_{int(dr*100)}pct_delta_{dstr}.npy")
        plt.figure(figsize=(4.8, 4.0))
        sns.heatmap(R, vmin=0, vmax=1, cmap='viridis', square=True, cbar_kws={'label': '$r_{ij}$'})
        plt.title(f'Final $r_{{ij}}$ Matrix ({int(dr*100)}% Defectors, $\\delta={delta}$)')
        plt.tight_layout()
        # safe filename for LaTeX
        plt.savefig(PAPER_DIR / f"rij_heatmap_{int(dr*100)}pct_delta_{dstr}.pdf")
        plt.close()
        print(f"Saved {PAPER_DIR / f'rij_heatmap_{int(dr*100)}pct_delta_{dstr}.pdf'}")
