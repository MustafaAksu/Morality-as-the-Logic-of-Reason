import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parametreler
N = 10  # Düğüm sayısı
t_steps = 100  # Zaman adımı sayısı
dt = 0.1  # Zaman adımı boyutu
kappa = 1.0  # Eğrilik sabiti
gamma = 1.0  # Faz sapma sabiti
k_B = 1.0  # Boltzmann sabiti
T_cog = 1.0  # Bilişsel sıcaklık
J_max = 1.0  # Maksimum bağ gücü
seeds = [42, 43, 44]  # Tekrarlar için seed'ler

# Faz diyagramı parametreleri
defect_ratios = np.arange(0.1, 0.61, 0.05)
forgiveness_levels = np.arange(0.1, 0.91, 0.1)
R_map = np.zeros((len(defect_ratios), len(forgiveness_levels)))

# Multi-cue için hafıza (son 10 adım)
coop_history = np.zeros((N, N, 10))  # Kooperasyon geçmişi
action_history = np.zeros((N, N, 5))  # Eylem istikrarı için
types = np.random.uniform(0, 1, N)  # Ajan tipleri (semantic için)

def compute_r_ij(t, i, j):
    # Multi-cue r_{ij} hesaplaması
    behavior = np.mean(coop_history[i, j, :])  # Son 10 adım i→j kooperasyon
    trust = np.mean(coop_history[j, i, :])  # Son 10 adım j→i kooperasyon
    semantic = 1 - abs(types[i] - types[j]) / 0.3  # Tip farkı
    semantic = np.clip(semantic, 0, 1)
    stability = 1 - np.std(action_history[i, j, :])  # Son 5 adım varyansı
    stability = np.clip(stability, 0, 1)
    r_ij = 0.4 * behavior + 0.3 * trust + 0.2 * semantic + 0.1 * stability
    return np.clip(r_ij, 0.1, 1.0)

# Faz diyagramı döngüsü
for di, defect_ratio in enumerate(defect_ratios):
    for fi, forgiveness in enumerate(forgiveness_levels):
        R_runs = []
        for seed in seeds:
            np.random.seed(seed)
            # Başlangıç durumları
            phi = np.random.uniform(0, 2 * np.pi, N)
            omega = np.random.uniform(-1, 1, N)
            J_ij = np.random.uniform(0.5, 0.8, (N, N))
            np.fill_diagonal(J_ij, 0)
            J_ij = (J_ij + J_ij.T) / 2
            coop_history[...] = 0.5  # Başlangıçta nötr kooperasyon
            action_history[...] = 0.5

            R_t = np.zeros(t_steps)
            E_e_t = np.zeros(t_steps)
            for t in range(t_steps):
                # Kuramoto dinamiği
                dphi = np.zeros(N)
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            dphi[i] += J_ij[i, j] * np.sin(phi[j] - phi[i])
                    dphi[i] += omega[i]
                phi += dphi * dt

                # Multi-cue J_ij güncellemesi
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            # Simüle kooperasyon (defektör oranı etkisi)
                            coop_prob = (1 - defect_ratio) * forgiveness
                            coop = np.random.rand() < coop_prob
                            coop_history[i, j, :-1] = coop_history[i, j, 1:]
                            coop_history[i, j, -1] = 1 if coop else 0
                            # Eylem istikrarı
                            action_history[i, j, :-1] = action_history[i, j, 1:]
                            action_history[i, j, -1] = coop_prob
                            # r_{ij} ve J_{ij}
                            r_ij = compute_r_ij(t, i, j)
                            J_ij[i, j] = r_ij * J_max  # J_ij ∝ r_{ij}
                J_ij = np.clip(J_ij, 0, J_max)
                J_ij = (J_ij + J_ij.T) / 2

                # Hesaplamalar
                J_avg = np.mean(J_ij[J_ij > 0])
                cos_delta = np.array([[np.cos(phi[i] - phi[j]) for j in range(N)] for i in range(N)])
                cos_avg = np.mean(np.abs(cos_delta[cos_delta != 0]))
                E_e = k_B * T_cog * np.log(J_max / J_avg) if J_avg < J_max else 0
                E_e_t[t] = E_e
                dE_dt = (E_e_t[t] - E_e_t[t-1]) / dt if t > 0 else 0
                sin_sq_sum = sum(J_ij[i, j] * np.sin(phi[i] - phi[j])**2 for i in range(N) for j in range(i+1, N))
                sin_sq_avg = sin_sq_sum / (N * (N-1) / 2) if N > 1 else 0
                R_t[t] = kappa * dE_dt + gamma * sin_sq_avg

            R_runs.append(np.mean(R_t))
        R_map[di, fi] = np.mean(R_runs)

# Faz diyagramı görselleştirme
plt.figure(figsize=(8, 6))
plt.imshow(R_map, origin='lower', extent=[forgiveness_levels[0], forgiveness_levels[-1], defect_ratios[0], defect_ratios[-1]], aspect='auto', cmap='viridis')
plt.colorbar(label='Ortalama R(t)')
plt.xlabel('Affedicilik Seviyesi')
plt.ylabel('Defektör Oranı')
plt.title('R(t) Faz Diyagramı')
plt.savefig('phase_R_map.pdf')
# plt.show()  # Yerel çalıştırırsan aç