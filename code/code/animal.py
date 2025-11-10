import numpy as np
import matplotlib.pyplot as plt

# ====================== 1. PARAMETERS ======================
# ---- core ----
alpha_f = 2.0;   alpha_E = 1.5;   alpha_phi = 5.0
beta_f  = 0.1;   beta = np.array([0.3, 0.2, 0.1])
Gamma   = np.array([0.8, 0.6, 0.4])
eta_E   = 0.05

# ---- potential ----
w_f = 1.0
W_E = np.diag([1.2, 0.8, 0.6])
lambda_phi = 3.0

# ---- setpoints & gates ----
f0 = 1.0
E0 = np.array([1.0, 0.7, 0.5])
theta_p, theta_c = 0.6, 0.4

# ---- OU base ----
tau_m = {'f':2.0, 'Ep':1.5, 'Ec':2.0, 'Er':3.0, 'Phi':1.0}
sigma_base = {'f':0.03,'Ep':0.04,'Ec':0.03,'Er':0.02,'Phi':0.05}

# ---- simulation ----
dt = 0.02
T  = 400
steps = int(T/dt)
rng = np.random.default_rng(42)

# ====================== 2. STATE INIT ======================
f   = f0
E   = E0.copy()
Phi = np.array([1.0, 0.0, 0.0])
xi  = {k:0.0 for k in tau_m}

# storage
hist = dict(f=[], E=[], Phi=[], V=[], g=[], I=[],
            f_env=[], s=[], g_pred=[], Vhat=[], traceSigma=[])

# ====================== 3. HELPERS ======================
def policy_dir(g):
    dirs = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    return np.array(dirs[g])

def project_tangent(v, phi):
    return v - np.dot(v, phi)*phi

# ====================== 4. CIL CLASS ======================
class CIL:
    def __init__(self, dt, d=4, alpha_mpc=2.0, curiosity_gamma=0.8):
        self.dt = dt
        self.d  = d
        self.M  = np.zeros(d)                 # [f_env, s_p, s_c, s_r]
        self.S  = np.eye(d) * 0.2
        self.A  = np.eye(d) * 0.98            # slight contraction → stable
        self.C  = np.eye(d)
        self.Q  = np.eye(d) * 1e-3
        self.R  = np.eye(d) * 5e-3
        self.alpha_mpc = alpha_mpc
        self.curiosity_gamma = curiosity_gamma   # trace(Sigma) → sigma boost
        # online drift learning buffers
        self.prev_obs = None
        self.ema_alpha = 0.001

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

        # ---- online drift learning (EWMA) ----
        if self.prev_obs is not None:
            delta = obs - self.prev_obs
            # least-squares direction: obs_t ≈ A * obs_{t-1}
            grad = np.outer(delta, self.prev_obs)
            self.A = (1-self.ema_alpha)*self.A + self.ema_alpha*grad
            # enforce spectral radius <1 for stability
            eig = np.linalg.eigvals(self.A)
            if np.any(np.abs(eig)>=1.0):
                self.A /= (1.01*np.max(np.abs(eig)))
        self.prev_obs = obs.copy()

        return self.M

    def forecast(self):
        return self.A @ self.M

    def uncertainty_boost(self):
        tr = np.trace(self.S)
        return 1.0 + self.curiosity_gamma * tr   # multiplicative factor

cil = CIL(dt, alpha_mpc=2.5, curiosity_gamma=0.9)

# ====================== 5. MAIN LOOP ======================
for step in range(steps):
    t = step*dt

    # ---- 5.1 Environment (four phases) ----
    if t < 50:                      # Rest
        f_env, s = f0, np.zeros(3)
    elif t < 200:                   # Hunger
        f_env = f0
        s = np.array([1.5, 0.0, 0.0])
    elif t < 240:                   # Threat (oscillating freq)
        f_env = f0 + 0.8*np.sin(12*t)
        s = np.array([0.3, 0.0, 0.0])
    else:                           # Play + curiosity spikes
        f_env = f0
        s = np.zeros(3)
        if (t-240) % 15 < dt*2 and rng.random()<0.7:
            xi['Phi'] += 0.45                     # big exploratory kick

    # ---- 5.2 CIL update ----
    obs = np.concatenate([[f_env], s])
    cil.correct(obs)
    fhat, sp, sc, sr = cil.forecast()
    shat = np.array([sp, sc, sr])
    e_hat = np.maximum(E0 - E, 0.0) + 0.2*shat
    g_pred = int(np.argmax(e_hat))
    Phi_star_pred = policy_dir(g_pred)

    # ---- 5.3 Gating (Maslow) ----
    I_p = 1.0
    I_c = 1.0 if E[0] > theta_p else 0.0
    I_r = 1.0 if (E[0] > theta_p and E[1] > theta_c) else 0.0
    I   = np.array([I_p, I_c, I_r])

    # ---- 5.4 Active channel (reactive) ----
    err = np.maximum(E0 - E, 0.0) + 0.2*s
    g   = int(np.argmax(err))
    u_g = np.eye(3)[g]
    Phi_star = policy_dir(g)

    # ---- 5.5 Curiosity-driven OU scaling ----
    boost = cil.uncertainty_boost()
    sigma = {k: sigma_base[k]*boost for k in sigma_base}

    # ---- 5.6 OU noise update ----
    for k in xi:
        tau, sig = tau_m[k], sigma[k]
        xi[k] += (-xi[k]/tau)*dt + sig*np.sqrt(2/tau)*np.sqrt(dt)*rng.normal()

    # ---- 5.7 Gradients from V (reactive) ----
    df_det = -2*alpha_f*w_f*(f - f_env)
    dE_det = -2*alpha_E*W_E@(E - E0)

    # ---- 5.8 Recovery (gated) + effort ----
    dE_rec = beta * I * (E0 - E)
    dE_eff = -Gamma * u_g

    # ---- 5.9 Orientation (reactive) ----
    mis = 1.0 - np.dot(Phi, Phi_star)
    grad_phi = 2*lambda_phi*mis*Phi_star
    dPhi_det = -alpha_phi * project_tangent(grad_phi, Phi)

    # ---- 5.10 MPC pre-rotation (predictive) ----
    mis_pred = 1.0 - np.dot(Phi, Phi_star_pred)
    grad_phi_pred = 2*lambda_phi*mis_pred*Phi_star_pred
    dPhi_mpc = -cil.alpha_mpc * project_tangent(grad_phi_pred, Phi)

    # ---- 5.11 Assemble increments ----
    df = (df_det + beta_f*(f0 - f) + eta_E*np.sum(E) + xi['f']) * dt
    dE = (dE_det + dE_rec + dE_eff) * dt + np.array([xi['Ep'],xi['Ec'],xi['Er']]) * dt
    dPhi = (dPhi_det + dPhi_mpc + xi['Phi']*rng.normal(size=3)) * dt

    # ---- 5.12 Integrate ----
    f   += df
    E   += dE
    E   = np.clip(E, 0.0, None)
    Phi += dPhi
    Phi /= np.linalg.norm(Phi)

    # ---- 5.13 Diagnostics ----
    V = w_f*(f-f_env)**2 + (E-E0)@W_E@(E-E0) + lambda_phi*(1-np.dot(Phi,Phi_star))**2
    Vhat = w_f*(f-fhat)**2 + (E-E0)@W_E@(E-E0) + lambda_phi*(1-np.dot(Phi,Phi_star_pred))**2

    # store
    hist['f'].append(f)
    hist['E'].append(E.copy())
    hist['Phi'].append(Phi.copy())
    hist['V'].append(V)
    hist['g'].append(g)
    hist['I'].append(I.copy())
    hist['f_env'].append(f_env)
    hist['s'].append(s.copy())
    hist['g_pred'].append(g_pred)
    hist['Vhat'].append(Vhat)
    hist['traceSigma'].append(np.trace(cil.S))

# ====================== 6. PLOTS ======================
tvec = np.arange(steps)*dt
Ehist = np.stack(hist['E'])

fig = plt.figure(figsize=(12,10))

# 1. Potential (actual vs. predicted)
ax1 = plt.subplot(4,1,1)
ax1.plot(tvec, hist['V'], label=r'$V(t)$ (actual)')
ax1.plot(tvec, hist['Vhat'], '--', label=r'$\hat V(t)$ (forecast)')
ax1.set_ylabel('Potential'); ax1.legend()
for tt in [50,200,240]: plt.axvline(tt, color='k', ls='--', alpha=0.3)

# 2. Energy tiers + gates
ax2 = plt.subplot(4,1,2)
ax2.plot(tvec, Ehist[:,0], label=r'$E_p$')
ax2.plot(tvec, Ehist[:,1], label=r'$E_c$')
ax2.plot(tvec, Ehist[:,2], label=r'$E_r$')
ax2.axhline(theta_p, color='gray', ls=':'); ax2.axhline(theta_c, color='gray', ls=':')
ax2.set_ylabel('Energy'); ax2.legend()

# 3. Active channel (reactive vs. predictive)
ax3 = plt.subplot(4,1,3)
ax3.plot(tvec, hist['g'], drawstyle='steps-post', label='g (reactive)')
ax3.plot(tvec, hist['g_pred'], '--', drawstyle='steps-post', label='g_pred (MPC)')
ax3.set_yticks([0,1,2]); ax3.set_yticklabels(['phys','cog','rel'])
ax3.set_ylabel('Channel'); ax3.legend()

# 4. Forecast uncertainty & curiosity boost
ax4 = plt.subplot(4,1,4)
ax4.plot(tvec, hist['traceSigma'], label=r'tr$(\Sigma)$')
ax4_twin = ax4.twinx()
boost = [1.0 + cil.curiosity_gamma*np.trace(cil.S) for _ in range(steps)]
ax4_twin.plot(tvec, boost, 'r--', alpha=0.6, label='OU boost')
ax4.set_xlabel('time (s)'); ax4.set_ylabel('Uncertainty')
ax4_twin.set_ylabel('Boost factor', color='r')
ax4.legend(loc='upper left'); ax4_twin.legend(loc='upper right')

plt.tight_layout()
plt.show()