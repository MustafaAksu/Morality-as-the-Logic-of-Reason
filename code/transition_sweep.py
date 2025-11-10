import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Parametreler
N_values = [10, 20, 40]  # Sistem boyutları
t_steps = 100  # Zaman adımı sayısı
dt = 0.1  # Zaman adımı boyutu
kappa = 1.0  # Eğrilik sabiti
gamma = 1.0  # Faz sapma sabiti
k_B = 1.0  # Boltzmann sabiti
T_cog = 1.0  # Bilişsel sıcaklık
J_max = 1.0  # Maksimum bağ gücü
forgiveness = 0.5  # Affedicilik seviyesi
seeds = list(range(42, 52))  # 10 seed
defect_ratios = np.arange(0.35, 0.56, 0.02)  # İnce tarama %35-%55
L_beh, L_trust, L_stab = 10, 10, 5  # Bellek uzunlukları

# Çıktılar için depolama
R_means = {N: [] for N in N_values}
R_cis = {N: [] for N in N_values}
SR_means = {N: [] for N in N_values}
SR_cis = {N: [] for N in N_values}
dRdp = {N: [] for N in N_values}
R_hyst_up = {N: [] for N in N_values}
R_hyst_down = {N: [] for N in N_values}

# Multi-cue r_ij hesaplaması
def compute_r_ij(t, i, j, coop_history, action_history, types):
    behavior = np.mean(coop_history[i, j, -L_beh:]) if t >= L_beh else np.mean(coop_history[i, j, :])
    trust = np.mean(coop_history[j, i, -L_trust:]) if t >= L_trust else np.mean(coop_history[j, i, :])
    semantic = 1 - abs(types[i] - types[j]) / 0.3
    semantic = np.clip(semantic, 0, 1)
    stability = 1 - np.std(action_history[i, j, -L_stab:]) if t >= L_stab else 1 - np.std(action_history[i, j, :])
    stability = np.clip(stability, 0, 1)
    r_ij = 0.4 * behavior + 0.3 * trust + 0.2 * semantic + 0.1 * stability
    return np.clip(r_ij, 0.1, 1.0)

# Simülasyon fonksiyonu
def run_simulation(N, defect_ratio, seed, direction='up'):
    np.random.seed(seed)
    phi = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.uniform(-1, 1, N)
    J_ij = np.random.uniform(0.5, 0.8, (N, N))
    np.fill_diagonal(J_ij, 0)
    J_ij = (J_ij + J_ij.T) / 2
    types = np.random.uniform(0, 1, N)
    coop_history = np.full((N, N, L_beh), 0.5)
    action_history = np.full((N, N, L_stab), 0.5)
    
    R_t = np.zeros(t_steps)
    SR_t = np.zeros(t_steps)
    
    for t in range(t_steps):
        # Kuramoto dinamiği
        dphi = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    dphi[i] += J_ij[i, j] * np.sin(phi[j] - phi[i])
            dphi[i] += omega[i]
        phi += dphi * dt
        
        # J_ij güncellemesi (multi-cue)
        for i in range(N):
            for j in range(N):
                if i != j:
                    coop_prob = (1 - defect_ratio) * forgiveness
                    coop = np.random.rand() < coop_prob
                    coop_history[i, j, :-1] = coop_history[i, j, 1:]
                    coop_history[i, j, -1] = 1 if coop else 0
                    action_history[i, j, :-1] = action_history[i, j, 1:]
                    action_history[i, j, -1] = coop_prob
                    r_ij = compute_r_ij(t, i, j, coop_history, action_history, types)
                    J_ij[i, j] = r_ij * J_max
            J_ij = np.clip(J_ij, 0, J_max)
            J_ij = (J_ij + J_ij.T) / 2
        
        # Hesaplamalar
        J_avg = np.mean(J_ij[J_ij > 0])
        cos_delta = np.array([[np.cos(phi[i] - phi[j]) for j in range(N)] for i in range(N)])
        cos_avg = np.mean(np.abs(cos_delta[cos_delta != 0]))
        E_e = k_B * T_cog * np.log(J_max / J_avg) if J_avg < J_max else 0
        dE_dt = (E_e - (R_t[t-1] if t > 0 else E_e)) / dt if t > 0 else 0
        sin_sq_sum = sum(J_ij[i, j] * np.sin(phi[i] - phi[j])**2 for i in range(N) for j in range(i+1, N))
        sin_sq_avg = sin_sq_sum / (N * (N-1) / 2) if N > 1 else 0
        R_t[t] = kappa * dE_dt + gamma * sin_sq_avg
        
        # SR (ilişkisel entropi)
        r_ij = J_ij / J_max
        SR = -np.sum(r_ij[r_ij > 0] * np.log(r_ij[r_ij > 0])) / (N * (N-1)) if np.any(r_ij > 0) else 0
        SR_t[t] = SR
    
    return np.mean(R_t), np.mean(SR_t)

# Ana döngü
for N in N_values:
    R_runs = {d: [] for d in defect_ratios}
    SR_runs = {d: [] for d in defect_ratios}
    R_runs_down = {d: [] for d in defect_ratios}
    
    # Yukarı tarama
    for defect_ratio in defect_ratios:
        for seed in seeds:
            R, SR = run_simulation(N, defect_ratio, seed, direction='up')
            R_runs[defect_ratio].append(R)
            SR_runs[defect_ratio].append(SR)
    
    # Aşağı tarama (histerezis)
    for defect_ratio in reversed(defect_ratios):
        for seed in seeds:
            R, _ = run_simulation(N, defect_ratio, seed, direction='down')
            R_runs_down[defect_ratio].append(R)
    
    # Ortalama ve CI
    for d in defect_ratios:
        R_means[N].append(np.mean(R_runs[d]))
        R_cis[N].append(1.96 * sem(R_runs[d]))
        SR_means[N].append(np.mean(SR_runs[d]))
        SR_cis[N].append(1.96 * sem(SR_runs[d]))
    
    # dR/dp (sayısal türev)
    R_vals = np.array(R_means[N])
    dRdp[N] = np.gradient(R_vals, defect_ratios)

# Grafikler
plt.figure(figsize=(10, 6))
for N in N_values:
    plt.errorbar(defect_ratios, R_means[N], yerr=R_cis[N], label=f'N={N}', capsize=3)
plt.xlabel('Defektör Oranı')
plt.ylabel('Ortalama R(t)')
plt.title('R(t) vs Defektör Oranı (Farklı Sistem Boyutları)')
plt.legend()
plt.savefig('phase_transition_R.pdf')
# plt.show()

plt.figure(figsize=(10, 6))
for N in N_values:
    plt.errorbar(defect_ratios, SR_means[N], yerr=SR_cis[N], label=f'N={N}', capsize=3)
plt.xlabel('Defektör Oranı')
plt.ylabel('Ortalama S^R')
plt.title('S^R vs Defektör Oranı (Farklı Sistem Boyutları)')
plt.legend()
plt.savefig('phase_transition_Sbar.pdf')
# plt.show()

plt.figure(figsize=(10, 6))
for N in N_values:
    plt.plot(defect_ratios, dRdp[N], label=f'N={N}')
plt.xlabel('Defektör Oranı')
plt.ylabel('dR/dp')
plt.title('R(t) Türev Zirvesi (Kritik Eşik)')
plt.legend()
plt.savefig('dRdp_peak.pdf')
# plt.show()

plt.figure(figsize=(10, 6))
for N in N_values:
    plt.plot(defect_ratios, R_means[N], label=f'N={N} (Yukarı)', linestyle='-')
    plt.plot(list(reversed(defect_ratios)), [np.mean(R_runs_down[d]) for d in reversed(defect_ratios)], 
             label=f'N={N} (Aşağı)', linestyle='--')
plt.xlabel('Defektör Oranı')
plt.ylabel('Ortalama R(t)')
plt.title('Histerezis Testi')
plt.legend()
plt.savefig('hysteresis_R.pdf')
# plt.show()

plt.figure(figsize=(10, 6))
for N in N_values:
    plt.plot(defect_ratios, R_means[N], label=f'N={N}')
plt.xlabel('Defektör Oranı')
plt.ylabel('Ortalama R(t)')
plt.title('Finite-Size Scaling: R(t) vs Defektör Oranı')
plt.legend()
plt.savefig('finite_size_transition.pdf')
# plt.show()