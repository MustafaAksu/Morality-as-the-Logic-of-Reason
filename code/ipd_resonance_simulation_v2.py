# Script Name: ipd_resonance_simulation_v4.py
# Purpose: Simulates a 10-agent Iterated Prisoner's Dilemma (IPD) to validate the "Morality as the Logic of Reason" framework.
#          Tests adaptive Generous Tit-for-Tat (GTFT) in minimizing relational entropy (S^R) under defector rates (20%, 50%) and
#          discount factors (δ = 0.5, 0.75, 0.95). Implements multi-cue resonance (r_ij) with weights [0.4, 0.3, 0.2, 0.1] for
#          behavior, semantics, trust, and stability, and uses ethical energy (E_c = (H-A)/(1+k t)^n) for decisions. Computes
#          normalized S^R (S^R_max = N(N-1)/e) for clarity. Parameters are grounded in literature (Axelrod 1984, Nowak 2006,
#          Frederick 2002, Santos et al. 2018) to minimize arbitrariness. Outputs cooperation rates, scores, S^R, and r_ij matrices
#          for figures (ipd_chart.pdf, entropy_chart.pdf, rij_heatmap_*.pdf) and GitHub transparency.
# Structure:
# - Initializes 10 agents with probabilistic defectors (20% or 50%).
# - Runs 200 IPD rounds with adaptive GTFT (forgive rate tuned by 10-round history).
# - Updates r_ij as a multi-cue blend (behavior, semantic type, trust, stability).
# - Uses E_c to guide decisions, estimating opponent actions from history.
# - Tracks S^R and normalized S^R per round for entropy minimization dynamics.
# - Tests δ = [0.5, 0.75, 0.95] to generate cooperation vs. δ plot.
# - Saves final r_ij for heatmaps to show relational isolation.
# - Saves results as CSV for LaTeX figures and GitHub (https://github.com/mustafa-aksu/morality-resonance).
# Authors: Mustafa Aksu, with AI collaborators (Grok by xAI, ChatGPT by OpenAI)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters (literature-grounded)
n_agents = 10           # Small network for demonstration (Axelrod 1984)
n_rounds = 200          # Sufficient for convergence (Axelrod 1984)
noise = 0.05            # Realistic error rate (Nowak 2006)
delta_values = [0.5, 0.75, 0.95]  # Discount factors (Frederick 2002)
k = 0.1                 # Impatience parameter (Frederick 2002)
n = 2                   # Quadratic discounting (Ainslie 1975)
forgive_base = 0.1      # Base forgive rate, tuned 0.05-0.15 (Santos et al. 2018)
defector_rates = [0.2, 0.5]  # Test fragility at 20% and 50%
w_k = [0.4, 0.3, 0.2, 0.1]  # Weights for r_ij: behavior, semantic, trust, stability
S_R_max = n_agents * (n_agents - 1) / np.e  # Max entropy for normalization

# Payoff matrix for IPD (C=1, D=0)
payoffs = {(1, 1): (3, 3), (1, 0): (0, 5), (0, 1): (5, 0), (0, 0): (1, 1)}

# Initialize agent types (semantic proximity: cooperative or adversarial)
np.random.seed(42)  # Reproducibility
agent_types = np.random.uniform(0.7, 1.0, n_agents)  # Cooperative bias (0.7-1.0)

# Adaptive GTFT strategy
def play_gtft(history, i, j, forgive_rate, noise):
    if np.random.random() < noise:
        return np.random.choice([0, 1])
    if len(history) == 0:
        return 1  # Cooperate first
    last_n = min(len(history), 10)  # Use last 10 rounds for adaptivity
    opp_actions = [history[t][j][i] for t in range(-last_n, 0)] if last_n > 0 else []
    trust = np.mean(opp_actions) if opp_actions else 1.0
    adaptive_forgive = forgive_base * (1 - trust)  # Higher forgive if less trust
    if np.random.random() < adaptive_forgive:
        return 1
    return history[-1][j][i] if history else 1

# E_c calculation for decision (H-A = payoff difference, t=1 for next round)
def compute_E_c(opp_prob, k=0.1, t=1, n=2):
    H_A = [(payoffs[(1, a)][0], 0) for a in [1, 0]]  # H = payoff, A = 0 (simplified cost)
    E_c_coop = sum(p * (h - a) / (1 + k * t) ** n for p, (h, a) in zip([opp_prob, 1 - opp_prob], H_A))
    E_c_defect = sum(p * (h - a) / (1 + k * t) ** n for p, (h, a) in zip([opp_prob, 1 - opp_prob], [(5, 0), (1, 1)]))
    return E_c_coop, E_c_defect

# Simulation results storage
results = []

# Run simulation for each defector rate and delta
for defector_rate in defector_rates:
    for delta in delta_values:
        print(f"\n--- Simulation with {defector_rate*100}% Defectors, δ={delta} ---")
        # Initialize defectors
        defectors = np.random.choice(n_agents, int(n_agents * defector_rate), replace=False)
        r_ij = np.random.uniform(0.5, 1.0, (n_agents, n_agents))  # Initial resonance
        np.fill_diagonal(r_ij, 1.0)  # Self-resonance = 1
        scores = np.zeros(n_agents)
        cooperation_rates = np.zeros(n_agents)
        history = []
        S_R_history = []
        S_R_bar_history = []

        # Run IPD rounds
        for round in range(n_rounds):
            round_actions = np.ones((n_agents, n_agents))
            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j:
                        if i in defectors and np.random.random() < 0.8:  # Probabilistic defection
                            round_actions[i, j] = 0
                        else:
                            # Estimate opponent's action probability
                            opp_actions = [history[t][j][i] for t in range(-min(len(history), 10), 0)] if history else []
                            opp_prob = np.mean(opp_actions) if opp_actions else 1.0
                            E_c_coop, E_c_defect = compute_E_c(opp_prob, k, t=1, n=2)
                            round_actions[i, j] = 1 if E_c_coop >= E_c_defect else play_gtft(history, i, j, forgive_base, noise)
            history.append(round_actions)

            # Update r_ij (multi-cue blend)
            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j:
                        behavior = np.mean([history[t][i][j] for t in range(-min(len(history), 10), 0)] if history else [1])
                        semantic = 1 - abs(agent_types[i] - agent_types[j]) / 0.3  # Normalize to [0,1]
                        trust = np.mean([history[t][j][i] for t in range(-min(len(history), 10), 0)] if history else [1])
                        stability = 1 - np.std([history[t][i][j] for t in range(-min(len(history), 5), 0)] if len(history) >= 5 else [1])
                        r_ij[i, j] = w_k[0] * behavior + w_k[1] * semantic + w_k[2] * trust + w_k[3] * stability
            # Compute S^R (un-normalized) and S^R_bar (normalized)
            S_R = -np.sum(r_ij * np.log(r_ij + 1e-10))
            S_R_bar = S_R / S_R_max
            S_R_history.append(S_R)
            S_R_bar_history.append(S_R_bar)
            # Update scores
            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j:
                        action_pair = (int(round_actions[i, j]), int(round_actions[j, i]))
                        scores[i] += payoffs[action_pair][0] * (delta ** round)
            cooperation_rates += np.mean(round_actions, axis=1)

        # Save final r_ij for heatmap
        np.save(f'final_rij_{int(defector_rate*100)}pct_delta_{delta}.npy', r_ij)

        # Final results
        cooperation_rates /= n_rounds
        avg_scores = scores / n_rounds
        avg_S_R = np.mean(S_R_history)
        avg_S_R_bar = np.mean(S_R_bar_history)
        results.append({
            'defector_rate': defector_rate,
            'delta': delta,
            'avg_scores': avg_scores.tolist(),
            'avg_coop_rates': cooperation_rates.tolist(),
            'avg_S_R': avg_S_R,
            'avg_S_R_bar': avg_S_R_bar
        })
        print(f"Average Scores: {avg_scores}")
        print(f"Cooperation Rates: {cooperation_rates}")
        print(f"Average Relational Entropy (S^R): {avg_S_R:.3f}")
        print(f"Normalized Relational Entropy (S^R_bar): {avg_S_R_bar:.3f}")

# Save results for plotting
pd.DataFrame(results).to_csv('simulation_results.csv')
print("Results saved to simulation_results.csv")

# Plot 1: Cooperation vs δ for 20% and 50% defectors
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for dr in defector_rates:
    coop_rates = [np.mean(r['avg_coop_rates']) for r in results if r['defector_rate'] == dr]
    deltas = [r['delta'] for r in results if r['defector_rate'] == dr]
    ax.plot(deltas, coop_rates, label=f'{dr*100}% Defectors', marker='o')
ax.set_xlabel('Discount Factor ($\\delta$)')
ax.set_ylabel('Mean Cooperation Rate')
ax.set_title('IPD: Cooperation Rate vs Discount Factor')
ax.legend()
plt.tight_layout()
plt.savefig('../paper/ipd_chart.pdf')
print('Saved ../paper/ipd_chart.pdf')

# Plot 2: S^R_bar over rounds for 20% and 50% defectors (δ=0.95)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for dr in defector_rates:
    for r in results:
        if r['defector_rate'] == dr and r['delta'] == 0.95:
            ax.plot(range(n_rounds), S_R_bar_history, label=f'{dr*100}% Defectors')
ax.set_xlabel('Round')
ax.set_ylabel('Normalized Relational Entropy ($\\bar{S}^{\\mathsf{R}}$)')
ax.set_title('Normalized Relational Entropy Over 200 Rounds ($\\delta=0.95$)')
ax.legend()
plt.tight_layout()
plt.savefig('../paper/entropy_chart.pdf')
print('Saved ../paper/entropy_chart.pdf')

# Plot 3: r_ij Heatmap for 20% and 50% defectors (δ=0.95)
for dr in defector_rates:
    for delta in delta_values:
        R = np.load(f'final_rij_{int(dr*100)}pct_delta_{delta}.npy')
        plt.figure(figsize=(4.8, 4.0))
        sns.heatmap(R, vmin=0, vmax=1, cmap='viridis', square=True, cbar_kws={'label': '$r_{ij}$'})
        plt.title(f'Final $r_{{ij}}$ Matrix ({int(dr*100)}% Defectors, $\\delta={delta}$)')
        plt.tight_layout()
        plt.savefig(f'../paper/rij_heatmap_{int(dr*100)}pct_delta_{delta}.pdf')
        print(f'Saved ../paper/rij_heatmap_{int(dr*100)}pct_delta_{delta}.pdf')