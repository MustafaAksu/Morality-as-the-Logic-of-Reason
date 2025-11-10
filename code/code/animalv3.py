import numpy as np
import matplotlib.pyplot as plt

# ====================== 1. PARAMETERS ======================
# Core dynamics
alpha_f = 2.0;   alpha_E = 1.5;   alpha_phi = 4.5
beta_f  = 0.1;   beta = np.array([0.3, 0.2, 0.1])
Gamma   = np.array([0.8, 0.6, 0.4])
eta_E   = 0.05

# Resonance potential
w_f = 1.0
W_E = np.diag([1.2, 0.8, 0.6])
lambda_phi = 3.0

# Setpoints & Maslow gates
f0 = 1.0
E0 = np.array([1.0, 0.7, 0.5])
theta_p, theta_c = 0.6, 0.4

# Base OU noise
tau_m = {'f':2.0, 'Ep':1.5, 'Ec':2.0, 'Er':3.0, 'Phi':1.0}
sigma_base = {'f':0.03, 'Ep':0.04, 'Ec':0.03, 'Er':0.02, 'Phi':0.05}

# Simulation
dt = 0.02
T  = 400
steps = int(T/dt)
rng = np.random.default_rng(42)

# ====================== 2. STATE INIT ======================
f   = f0
E   = E0.copy()
Phi = np.array([1.0, 0.0, 0.0])
xi  = {k:0.0 for k in tau_m}

# Storage
hist = {
    'f':[], 'E':[], 'Phi':[], 'V':[], 'Vhat':[], 'g':[], 'g_pred':[], 'I':[],
    'f_env':[], 's':[], 'traceSigma':[], 'boost':[]
}

# ====================== 3. HELPERS ======================
def policy_dir(g):
    dirs = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    return np.array(dirs[g])

def project_tangent(v, phi):
    return v - np.dot(v, phi) * phi

# ====================== 4. CIL CLASS (with learning + curiosity) ======================
class CIL:
    def __init__(self, dt, d=4, alpha_mpc=3.0, curiosity_gamma=1.2, ema_rho=0.002):
        self.dt = dt
        self.d  = d
        self.M  = np.zeros(d)
        self.S  = np.eye(d) * 0.2
        self.A  = np.eye(d) * 0.98
        self.C  = np.eye(d)
        self.Q  = np.eye(d) * 1e-3
        self.R  = np.eye(d) * 5e-3
        self.alpha_mpc = alpha_mpc
        self.curiosity_gamma = curiosity_gamma
        self.ema_rho = ema_rho
        self.prev_obs = None

    def predict(self):
        M_ = self.A @ self.M
        S_ = self.A @ self.S @ self.A.T + self.Q
        return M_, S_

    def correct(self, obs):
        M_, S_ = self.predict()
        H = self.C
        Syy = H @ S_ @ H.T + self.R
        K = S_ @ H.T @ np.linalg.inv(Syy)
        innov = obs - H @ M_
        self.M = M_ + K @ innov
        self.S = (np.eye(self.d) - K @ H) @ S_

        # ---- Online drift learning (stable EWMA) ----
        if self.prev_obs is not None:
            delta = obs - self.prev_obs
            grad = np.outer(delta, self.prev_obs)
            self.A = (1 - self.ema_rho) * self.A + self.ema_rho * grad
            # Clamp spectral radius
            eig = np.linalg.eigvals(self.A)
            if np.any(np.abs(eig) >= 1.0):
                self.A /= (1.01 * np.max(np.abs(eig)))
        self.prev_obs = obs.copy()

        return self.M

    def forecast(self):
        return self.A @ self.M

    def uncertainty_boost_vector(self):
        Sdiag = np.clip(np.diag(self.S), 0, None)
        return 1.0 + self.curiosity_gamma * Sdiag  # [f, sp, sc, sr]

cil = CIL(dt, alpha_mpc=3.0, curiosity_gamma=1.2, ema_rho=0.002)

# ====================== 5. MAIN LOOP ======================
for step in range(steps):
    t = step * dt

    # ---- 5.1 Environment (predictable + tier switch) ----
    base_osc = 0.10 * np.sin(0.25 * t)  # Learnable slow rhythm
    if t < 50:                      # Rest
        f_env = f0 + base_osc
        s = np.zeros(3)
    elif t < 200:                   # Hunger
        f_env = f0 + base_osc
        s = np.array([1.5, 0.0, 0.0])
    elif t < 240:                   # Threat
        f_env = f0 + 0.8 * np.sin(12 * t)
        s = np.array([0.3, 0.0, 0.0])
    else:                           # Play + Cognitive burst after recovery
        f_env = f0 + base_osc
        s = np.array([0.0, 1.2 if t > 280 else 0.0, 0.0])
        if (t - 240) % 15 < dt * 2 and rng.random() < 0.7:
            xi['Phi'] += 0.45

    # ---- 5.2 CIL update ----
    obs = np.concatenate([[f_env], s])
    cil.correct(obs)
    fhat, sp, sc, sr = cil.forecast()
    shat = np.array([sp, sc, sr])

    # ---- 5.3 Per-component curiosity boost ----
    boost_vec = cil.uncertainty_boost_vector()
    sigma = {
        'f':   sigma_base['f']   * boost_vec[0],
        'Ep':  sigma_base['Ep']  * boost_vec[1],
        'Ec':  sigma_base['Ec']  * boost_vec[2],
        'Er':  sigma_base['Er']  * boost_vec[3],
        'Phi': sigma_base['Phi'] * (0.5 * boost_vec[1] + 0.5 * boost_vec[2])
    }

    # ---- 5.4 Gating (Maslow) ----
    I_p = 1.0
    I_c = 1.0 if E[0] > theta_p else 0.0
    I_r = 1.0 if (E[0] > theta_p and E[1] > theta_c) else 0.0
    I = np.array([I_p, I_c, I_r])

    # ---- 5.5 Active channel (reactive) ----
    err = np.maximum(E0 - E, 0.0) + 0.2 * s
    g = int(np.argmax(err))
    u_g = np.eye(3)[g]
    Phi_star = policy_dir(g)

    # ---- 5.6 Predicted channel (MPC) ----
    e_hat = np.maximum(E0 - E, 0.0) + 0.2 * shat
    g_pred = int(np.argmax(e_hat))
    Phi_star_pred = policy_dir(g_pred)

    # ---- 5.7 OU noise update ----
    for k in xi:
        tau, sig = tau_m[k], sigma[k]
        xi[k] += (-xi[k]/tau) * dt + sig * np.sqrt(2/tau) * np.sqrt(dt) * rng.normal()

    # ---- 5.8 Gradients from V (reactive) ----
    df_det = -2 * alpha_f * w_f * (f - f_env)
    dE_det = -2 * alpha_E * W_E @ (E - E0)

    # ---- 5.9 Recovery + effort ----
    dE_rec = beta * I * (E0 - E)
    dE_eff = -Gamma * u_g

    # ---- 5.10 Orientation: reactive + MPC ----
    mis = 1.0 - np.dot(Phi, Phi_star)
    grad_phi = 2 * lambda_phi * mis * Phi_star
    dPhi_det = -alpha_phi * project_tangent(grad_phi, Phi)

    mis_pred = 1.0 - np.dot(Phi, Phi_star_pred)
    grad_phi_pred = 2 * lambda_phi * mis_pred * Phi_star_pred
    dPhi_mpc = -cil.alpha_mpc * project_tangent(grad_phi_pred, Phi)

    # ---- 5.11 Assemble increments ----
    df = (df_det + beta_f * (f0 - f) + eta_E * np.sum(E) + xi['f']) * dt
    dE = (dE_det + dE_rec + dE_eff) * dt + np.array([xi['Ep'], xi['Ec'], xi['Er']]) * dt
    dPhi = (dPhi_det + dPhi_mpc + xi['Phi'] * rng.normal(size=3)) * dt

    # ---- 5.12 Integrate ----
    f += df
    E += dE
    E = np.clip(E, 0.0, None)
    Phi += dPhi
    Phi /= np.linalg.norm(Phi)

    # ---- 5.13 Diagnostics ----
    V = w_f * (f - f_env)**2 + (E - E0) @ W_E @ (E - E0) + lambda_phi * (1 - np.dot(Phi, Phi_star))**2
    Vhat = w_f * (f - fhat)**2 + (E - E0) @ W_E @ (E - E0) + lambda_phi * (1 - np.dot(Phi, Phi_star_pred))**2

    # Store
    hist['f'].append(f)
    hist['E'].append(E.copy())
    hist['Phi'].append(Phi.copy())
    hist['V'].append(V)
    hist['Vhat'].append(Vhat)
    hist['g'].append(g)
    hist['g_pred'].append(g_pred)
    hist['I'].append(I.copy())
    hist['f_env'].append(f_env)
    hist['s'].append(s.copy())
    hist['traceSigma'].append(np.trace(cil.S))
    hist['boost'].append(boost_vec.copy())

# ====================== 6. METRIC: MPC Lead Time ======================
g = np.array(hist['g'])
gpred = np.array(hist['g_pred'])
flip = np.diff(g, prepend=g[0]) != 0
flipp = np.diff(gpred, prepend=gpred[0]) != 0
lead_samples = []
for i in np.where(flip)[0]:
    j = np.where(flipp[:i])[0]
    if len(j): lead_samples.append(i - j[-1])
lead_time = np.mean(np.array(lead_samples) * dt) if lead_samples else np.nan
print(f"\nMean MPC lead time: {lead_time:.3f} s")

# ====================== 7. PLOTS ======================
tvec = np.arange(steps) * dt
Ehist = np.stack(hist['E'])

fig = plt.figure(figsize=(12, 11))

# 1. Potential
ax1 = plt.subplot(5,1,1)
ax1.plot(tvec, hist['V'], label=r'$V(t)$')
ax1.plot(tvec, hist['Vhat'], '--', label=r'$\hat V(t)$ (MPC)')
ax1.set_ylabel('Potential'); ax1.legend()
for tt in [50,200,240,280]: plt.axvline(tt, color='k', ls='--', alpha=0.3)

# 2. Energy
ax2 = plt.subplot(5,1,2)
ax2.plot(tvec, Ehist[:,0], label=r'$E_p$')
ax2.plot(tvec, Ehist[:,1], label=r'$E_c$')
ax2.plot(tvec, Ehist[:,2], label=r'$E_r$')
ax2.axhline(theta_p, color='gray', ls=':'); ax2.axhline(theta_c, color='gray', ls=':')
ax2.set_ylabel('Energy'); ax2.legend()

# 3. Channels
ax3 = plt.subplot(5,1,3)
ax3.plot(tvec, hist['g'], drawstyle='steps-post', lw=2, label='g (reactive)')
ax3.plot(tvec, hist['g_pred'], '--', drawstyle='steps-post', lw=2, label='g_pred (MPC)')
ax3.set_yticks([0,1,2]); ax3.set_yticklabels(['phys','cog','rel'])
ax3.set_ylabel('Channel'); ax3.legend()

# 4. Uncertainty & Curiosity Boost
ax4 = plt.subplot(5,1,4)
ax4.plot(tvec, hist['traceSigma'], label=r'tr$(\Sigma)$')
ax4_twin = ax4.twinx()
boost_avg = [np.mean(b) for b in hist['boost']]
ax4_twin.plot(tvec, boost_avg, 'r--', alpha=0.7, label='Avg OU boost')
ax4.set_ylabel('Uncertainty'); ax4_twin.set_ylabel('Boost', color='r')
ax4.legend(loc='upper left'); ax4_twin.legend(loc='upper right')

# 5. Orientation (example: x-component)
ax5 = plt.subplot(5,1,5)
Phi_x = [p[0] for p in hist['Phi']]
ax5.plot(tvec, Phi_x, label=r'$\Phi_x$')
ax5.set_xlabel('time (s)'); ax5.set_ylabel(r'$\Phi_x$'); ax5.legend()

plt.tight_layout()
plt.show()