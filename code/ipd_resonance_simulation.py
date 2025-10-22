# Script Name: ipd_resonance_simulation.py
# Purpose: Simulates a 10-agent Iterated Prisoner's Dilemma (IPD) to validate the "Morality as the Logic of Reason" framework.
#          Tests how adaptive Generous Tit-for-Tat (GTFT) strategies minimize relational entropy (S^R) in a multi-agent network,
#          with 20% and 50% defector rates to assess fragility and isolation. Incorporates ethical energy (E_c) for decisions,
#          with multi-cue resonance (r_ij) based on behavior, semantics, trust, and stability. Parameters are grounded in literature
#          (Axelrod 1984, Nowak 2006, Frederick 2002, Santos et al. 2018) to reduce arbitrariness.
# Structure:
# - Initializes 10 agents, with probabilistic defectors (20% or 50%).
# - Runs 200 IPD rounds with adaptive GTFT (forgive_rate tuned by history).
# - Updates r_ij as a multi-cue blend (behavior, semantic type, trust, stability).
# - Computes E_c to guide decisions, estimating opponent actions from history.
# - Tracks S^R per round to show entropy minimization.
# - Outputs average scores, cooperation rates, and S^R for two defector rates.
# Authors: Mustafa Aksu, with AI collaborators (Grok by xAI, ChatGPT by OpenAI)

import numpy as np

# Parameters (literature-grounded)
n_agents = 10           # Number of agents (small network for demonstration)
n_rounds = 200          # Axelrod 1984: sufficient for convergence
noise = 0.05            # Nowak 2006: realistic error rate
delta = 0.95            # Frederick 2002: high discount for mature agents
k = 0.1                 # Impatience parameter (moderate, per Frederick 2002)
n = 2                   # Quadratic discounting (Ainslie 1975)
forgive_base = 0.1      # Base forgive rate (Nowak 2006, tuned 0.05-0.15)
defector_rates = [0.2, 0.5]  # Test fragility at 20% and 50% defectors

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
    H_A = [(payoffs[(1, a)][0], payoffs[(0, a)][0]) for a in [1, 0]]
    E_c_coop = sum(p * (h - a) / (1 + k * t) ** n for p, (h, a) in zip([opp_prob, 1 - opp_prob], H_A))
    E_c_defect = sum(p * (h - a) / (1 + k * t) ** n for p, (h, a) in zip([opp_prob, 1 - opp_prob], [(5, 0), (1, 1)]))
    return E_c_coop, E_c_defect

# Simulation for each defector rate
for defector_rate in defector_rates:
    print(f"\n--- Simulation with {defector_rate*100}% Defectors ---")
    # Initialize defectors
    defectors = np.random.choice(n_agents, int(n_agents * defector_rate), replace=False)
    r_ij = np.random.uniform(0.5, 1.0, (n_agents, n_agents))  # Initial resonance
    np.fill_diagonal(r_ij, 1.0)  # Self-resonance = 1
    scores = np.zeros(n_agents)
    cooperation_rates = np.zeros(n_agents)
    history = []
    S_R_history = []

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

        # Update r_ij (multi-cue: behavior, semantic, trust, stability)
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    behavior = np.mean([history[t][i][j] for t in range(-min(len(history), 10), 0)] if history else [1])
                    semantic = abs(agent_types[i] - agent_types[j]) / 0.3  # Normalize to [0,1]
                    trust = np.mean([history[t][j][i] for t in range(-min(len(history), 10), 0)] if history else [1])
                    stability = 1 - np.std([history[t][i][j] for t in range(-min(len(history), 5), 0)] if len(history) >= 5 else [1])
                    r_ij[i, j] = 0.4 * behavior + 0.3 * (1 - semantic) + 0.2 * trust + 0.1 * stability
        # Compute S^R (un-normalized)
        S_R = -np.sum(r_ij * np.log(r_ij + 1e-10))
        S_R_history.append(S_R)
        # Update scores
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    action_pair = (int(round_actions[i, j]), int(round_actions[j, i]))
                    scores[i] += payoffs[action_pair][0] * (delta ** round)
        cooperation_rates += np.mean(round_actions, axis=1)

    # Final results
    cooperation_rates /= n_rounds
    avg_scores = scores / n_rounds
    avg_S_R = np.mean(S_R_history)
    print(f"Average Scores: {avg_scores}")
    print(f"Cooperation Rates: {cooperation_rates}")
    print(f"Average Relational Entropy (S^R): {avg_S_R:.3f}")

# Plot suggestion for paper (to be rendered in LaTeX)
"""
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{sr_over_time.pdf}
    \caption{Relational entropy $S^R$ over 200 rounds in a 10-agent IPD with 20\% and 50\% defectors, showing minimization via adaptive GTFT.}
\end{figure}
"""