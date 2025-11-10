import numpy as np, matplotlib.pyplot as plt, pandas as pd
np.random.seed(7)

initial, months, paths = 10_000, 60, 10_000   # 5 yıl
risk_scale = 3.0   # Half–Full arası temkinli ölçek

# Rejimler: 0=boğa, 1=yatay, 2=ayı
mu  = np.array([0.030, 0.010, -0.020]) * risk_scale
sig = np.array([0.060, 0.030, 0.070]) * risk_scale
P = np.array([[0.80,0.10,0.10],
              [0.20,0.60,0.20],
              [0.25,0.25,0.50]])

def simulate_regimes(n, T, P, start=1):
    s = np.zeros((n,T), dtype=int); s[:,0]=start
    cP = np.cumsum(P,axis=1)
    for t in range(1,T):
        u = np.random.rand(n); prev = s[:,t-1]; nxt = np.empty(n,int)
        for st in (0,1,2):
            idx=(prev==st)
            if idx.any():
                uu=u[idx][:,None]; cp=cP[st][None,:]
                nxt[idx]=(uu<=cp).argmax(1)
        s[:,t]=nxt
    return s

states = simulate_regimes(paths, months, P, start=1)
r = np.random.normal(mu[states], sig[states])
r = np.clip(r, -0.90, None)  # tek ayda -%100 altını engelle

W = initial * np.cumprod(1+r, axis=1)
p05, p50, p95 = np.percentile(W, [5,50,95], axis=0)

# Bant grafiği
plt.figure(figsize=(10,6))
plt.plot(p50, label='Medyan (P50)')
plt.plot(p05, label='P5')
plt.plot(p95, label='P95')
plt.plot(W[np.random.randint(0,paths)], label='Örnek patika', alpha=0.7)
plt.xlabel('Ay'); plt.ylabel('Sermaye (TL)')
plt.title('Regime-Switching Monte Carlo (risk_scale=%.2f)'%risk_scale)
plt.legend(); plt.tight_layout(); plt.show()

# 5. yıl sonu log-histogram
plt.figure(figsize=(9,5))
bins = np.logspace(np.log10(max(W[:,-1].min(),1)), np.log10(W[:,-1].max()), 80)
plt.hist(W[:,-1], bins=bins, density=True, alpha=0.6)
plt.xscale('log'); plt.xlabel('5. Yıl Sonu Sermaye (TL, log ölçek)'); plt.ylabel('Yoğunluk')
plt.title('5 Yıl Sonu Sermaye Dağılımı (risk_scale=%.2f)'%risk_scale)
plt.tight_layout(); plt.show()

# 12 ve 60. ay özetleri:
for m in (12,60):
    P = np.percentile(W[:,m-1],[5,50,95,98])
    print(f"{m}. ay  P5={P[0]:.0f}, Medyan={P[1]:.0f}, P95={P[2]:.0f}, P98={P[3]:.0f}")
