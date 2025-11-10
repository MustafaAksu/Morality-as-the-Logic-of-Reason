# animalv45.py
import numpy as np
import matplotlib.pyplot as plt

# ====================== 0) CONFIG ======================
PRINT_DEBUG_FLIPS   = False
PRED_HORIZON_S      = 2.5      # seconds to look ahead
PRED_KAPPA          = 0.20     # reactive stimulus weight
PRED_KAPPA_PRED     = 0.20     # predicted stimulus weight
PRED_MARGIN         = 0.02     # minimum e_hat gap to accept a predicted switch
STABLE_S            = 1.0      # flip must persist this long to count
RAMP_START, RAMP_END = 260.0, 300.0
SEED = 42

# ====================== 1) PARAMETERS ======================
alpha_f = 2.0;   alpha_E = 1.5;   alpha_phi = 3.5
beta_f  = 0.1;   beta = np.array([0.30, 0.20, 0.10])
Gamma   = np.array([0.80, 0.60, 0.40])
eta_E   = 0.05

w_f = 1.0
W_E = np.diag([1.2, 0.8, 0.6])
lambda_phi = 3.0

f0 = 1.0
E0 = np.array([1.0, 0.7, 0.5])

# Hysteresis thresholds
theta_p_on, theta_p_off = 0.62, 0.58
theta_c_on, theta_c_off = 0.42, 0.38

# OU noise base
tau_m = {'f':2.0, 'Ep':1.5, 'Ec':2.0, 'Er':3.0, 'Phi':1.0}
sigma_base = {'f':0.03, 'Ep':0.04, 'Ec':0.03, 'Er':0.02, 'Phi':0.05}

# Simulation
dt = 0.02
T  = 400
steps = int(T/dt)
rng = np.random.default_rng(SEED)

# ====================== 2) STATE INIT ======================
f   = f0
E   = E0.copy()
Phi = np.array([1.0, 0.0, 0.0])
xi  = {k:0.0 for k in tau_m}

gate_p_open = True
gate_c_open = False

hist = {k:[] for k in [
    'f','E','Phi','V','Vhat','g','g_pred','I','f_env','s','traceSigma','boost_phi'
]}

# ====================== 3) HELPERS ======================
def policy_dir(g):
    dirs = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    return np.array(dirs[g])

def project_tangent(v, phi):
    return v - np.dot(v, phi) * phi

def first_stable_index(a, value, stable_samples, start_idx=0, end_idx=None):
    """First index where a==value holds for 'stable_samples' consecutive steps."""
    if end_idx is None: end_idx = len(a)
    c = 0
    for i in range(start_idx, end_idx):
        if a[i] == value:
            c += 1
            if c >= stable_samples:
                return i - stable_samples + 1
        else:
            c = 0
    return None

# ---- long-horizon rollout for (E,f) with predicted stimuli and gates ----
def predict_state_ahead(E_cur, f_cur, shat, horizon_s, dt_outer, params, gate_state):
    alpha_f, alpha_E, beta, Gamma, W_E, w_f, f0, E0 = params
    dt_h = dt_outer / 5.0
    steps = max(1, int(horizon_s / dt_h))

    Eh = E_cur.copy()
    fh = f_cur
    p_open, c_open = gate_state['p'], gate_state['c']

    for _ in range(steps):
        # hysteresis on predicted E
        if p_open and Eh[0] < theta_p_off: p_open = False
        elif (not p_open) and Eh[0] > theta_p_on: p_open = True

        if p_open and c_open and Eh[1] < theta_c_off: c_open = False
        elif p_open and (not c_open) and Eh[1] > theta_c_on: c_open = True

        I_pred = np.array([1.0, 1.0 if c_open else 0.0, 0.0])

        e_hat = np.maximum(E0 - Eh, 0.0) + PRED_KAPPA_PRED * shat
        # choose active channel; tie-break gently toward cog
        if abs(e_hat[1] - e_hat[0]) <= 0.01 * (1.0 + e_hat.max()):
            e_hat[1] += 1e-6
        g_hat = int(np.argmax(e_hat))
        u_hat = np.eye(3)[g_hat]

        # deterministic drifts (hold env at f0 during the short rollout)
        df_det = -2*alpha_f*w_f*(fh - f0)
        dE_det = -2*alpha_E*W_E @ (Eh - E0)
        dE_rec = beta * I_pred * (E0 - Eh)
        dE_eff = -Gamma * u_hat

        fh += df_det * dt_h
        Eh += (dE_det + dE_rec + dE_eff) * dt_h
        Eh = np.clip(Eh, 0.0, None)

    return Eh, fh, p_open, c_open

# ====================== 4) CIL (predictive layer) ======================
class CIL:
    def __init__(self, dt, d=4, alpha_mpc=4.5, curiosity_gamma=1.6, ema_rho=0.002):
        self.dt = dt; self.d = d
        self.M = np.zeros(d)
        self.S = np.eye(d) * 0.2
        self.A = np.eye(d) * 0.98
        self.C = np.eye(d)
        self.Q = np.eye(d) * 5e-3
        self.R = np.eye(d) * 8e-3
        self.alpha_mpc = alpha_mpc
        self.curiosity_gamma = curiosity_gamma
        self.ema_rho = ema_rho
        self.prev_obs = None
        self.prev_M = None

        # weak cross-coupling
        self.A += np.array([
            [0.00, 0.03, 0.00, 0.00],
            [0.02, 0.00, 0.04, 0.00],
            [0.00, 0.03, 0.00, 0.02],
            [0.00, 0.00, 0.02, 0.00],
        ])
        self.A *= 0.98

    def predict(self):
        M_ = self.A @ self.M
        S_ = self.A @ self.S @ self.A.T + self.Q
        return M_, S_

    def correct(self, obs):
        M_prev_for_vel = self.M.copy()
        M_, S_ = self.predict()
        H = self.C; Syy = H @ S_ @ H.T + self.R
        K = S_ @ H.T @ np.linalg.inv(Syy)
        innov = obs - H @ M_
        self.M = M_ + K @ innov
        self.S = (np.eye(self.d) - K @ H) @ S_
        if self.prev_obs is not None:
            delta = obs - self.prev_obs
            grad = np.outer(delta, self.prev_obs)
            self.A = (1 - self.ema_rho) * self.A + self.ema_rho * grad
            eig = np.linalg.eigvals(self.A)
            if np.any(np.abs(eig) >= 1.0):
                self.A /= (1.01 * np.max(np.abs(eig)))
        self.prev_obs = obs.copy()
        self.prev_M = M_prev_for_vel
        return self.M

    def forecast(self):
        return self.A @ self.M

    def forecast_seconds(self, seconds, dt):
        if self.prev_M is None:
            return self.forecast()
        v = (self.M - self.prev_M) / max(dt, 1e-9)
        return self.M + seconds * v

cil = CIL(dt, alpha_mpc=4.5, curiosity_gamma=1.6, ema_rho=0.002)

# ====================== 5) MAIN LOOP ======================
for step in range(steps):
    t = step * dt

    # ---- Environment ----
    base_osc = 0.10 * np.sin(0.25 * t)
    if t < 50:
        f_env = f0 + base_osc; s = np.zeros(3)
    elif t < 200:
        f_env = f0 + base_osc; s = np.array([1.5, 0.0, 0.0])
    elif t < 240:
        f_env = f0 + 0.8 * np.sin(12 * t); s = np.array([0.3, 0.0, 0.0])
    else:
        f_env = f0 + base_osc
        ramp = np.clip((t - RAMP_START) / (RAMP_END - RAMP_START), 0.0, 1.0)
        s = np.array([0.0, 1.2 * ramp, 0.0])
        if (t - 240) % 15 < dt * 2 and rng.random() < 0.7:
            xi['Phi'] += 0.45

    # ---- CIL posterior update ----
    obs = np.concatenate([[f_env], s])
    cil.correct(obs)

    # ---- future stimuli via constant-velocity forecast ----
    Mh = cil.forecast_seconds(seconds=PRED_HORIZON_S, dt=dt)
    fhat, sp, sc, sr = Mh
    shat = np.array([sp, sc, sr])

    # ---- long-horizon rollout for (E,f) using shat ----
    params = (alpha_f, alpha_E, beta, Gamma, W_E, w_f, f0, E0)
    gate_init = {'p': gate_p_open, 'c': gate_c_open}
    E_hat_pred, f_hat_pred, p_pred, c_pred = predict_state_ahead(
        E, f, shat, horizon_s=PRED_HORIZON_S, dt_outer=dt, params=params, gate_state=gate_init
    )

    # ---- PREDICTED channel (only if future cognitive gate will be open & advantage > margin) ----
    e_hat = np.maximum(E0 - E_hat_pred, 0.0) + PRED_KAPPA_PRED * shat
    # compute gap vs best non-cognitive option
    alt = max(e_hat[0], e_hat[2])
    advantage = e_hat[1] - alt
    if c_pred and (advantage > PRED_MARGIN):
        g_pred = 1
    else:
        g_pred = 0
    Phi_star_pred = policy_dir(g_pred)

    # ---- CURRENT hysteresis gating (recovery) ----
    if gate_p_open and E[0] < theta_p_off: gate_p_open = False
    elif (not gate_p_open) and E[0] > theta_p_on: gate_p_open = True

    if gate_p_open and gate_c_open and E[1] < theta_c_off: gate_c_open = False
    elif gate_p_open and (not gate_c_open) and E[1] > theta_c_on: gate_c_open = True

    I = np.array([1.0, 1.0 if gate_c_open else 0.0, 0.0])

    # ---- Reactive channel ----
    err = np.maximum(E0 - E, 0.0) + PRED_KAPPA * s
    g = int(np.argmax(err))
    u_g = np.eye(3)[g]
    Phi_star = policy_dir(g)

    # ---- Curiosity / OU scaling ----
    Sdiag = np.clip(np.diag(cil.S), 0, None)
    phi_boost = 1.0 + 1.2 * (0.5 * Sdiag[1] + 0.5 * Sdiag[2])
    sigma = {
        'f':   sigma_base['f']   * (1.0 + cil.curiosity_gamma * Sdiag[0]),
        'Ep':  sigma_base['Ep']  * (1.0 + cil.curiosity_gamma * Sdiag[1]),
        'Ec':  sigma_base['Ec']  * (1.0 + cil.curiosity_gamma * Sdiag[2]),
        'Er':  sigma_base['Er']  * (1.0 + cil.curiosity_gamma * Sdiag[3]),
        'Phi': sigma_base['Phi'] * phi_boost
    }
    for k in xi:
        tau, sig = tau_m[k], sigma[k]
        xi[k] += (-xi[k]/tau) * dt + sig * np.sqrt(2/tau) * np.sqrt(dt) * rng.normal()

    # ---- Deterministic dynamics ----
    df_det = -2 * alpha_f * w_f * (f - f_env)
    dE_det = -2 * alpha_E * W_E @ (E - E0)
    dE_rec = beta * I * (E0 - E)
    dE_eff = -Gamma * u_g

    mis = 1.0 - np.dot(Phi, Phi_star)
    dPhi_det = -alpha_phi * project_tangent(2 * lambda_phi * mis * Phi_star, Phi)

    mis_pred = 1.0 - np.dot(Phi, Phi_star_pred)
    dPhi_mpc  = -cil.alpha_mpc * project_tangent(2 * lambda_phi * mis_pred * Phi_star_pred, Phi)

    # integrate
    f   += (df_det + beta_f * (f0 - f) + eta_E * np.sum(E) + xi['f']) * dt
    E   += (dE_det + dE_rec + dE_eff) * dt + np.array([xi['Ep'], xi['Ec'], xi['Er']]) * dt
    E    = np.clip(E, 0.0, None)
    Phi += (dPhi_det + dPhi_mpc + xi['Phi'] * rng.normal(size=3)) * dt
    Phi /= np.linalg.norm(Phi)

    # ---- Diagnostics (Vhat uses predicted ENERGY) ----
    V = w_f * (f - f_env)**2 + (E - E0) @ W_E @ (E - E0) + lambda_phi * (1 - np.dot(Phi, Phi_star))**2
    Vhat = (
        w_f * (f - f_hat_pred)**2
        + (E_hat_pred - E0) @ W_E @ (E_hat_pred - E0)
        + lambda_phi * (1 - np.dot(Phi, Phi_star_pred))**2
    )

    if PRINT_DEBUG_FLIPS and len(hist['g']) > 0:
        if g != hist['g'][-1]:
            print(f"t={t:6.2f}s  REACTIVE flip -> {g},  Ep={E[0]:.3f}, Ec={E[1]:.3f}")
        if g_pred != (hist['g_pred'][-1] if hist['g_pred'] else g_pred):
            print(f"t={t:6.2f}s  PREDICT  flip -> {g_pred}, Ep_hat={E_hat_pred[0]:.3f}, Ec_hat={E_hat_pred[1]:.3f}")

    for k, v in zip(hist.keys(), [f,E,Phi,V,Vhat,g,g_pred,I,f_env,s,np.trace(cil.S),phi_boost]):
        hist[k].append(v if np.isscalar(v) else v.copy())

# ====================== 6) METRICS ======================
tvec = np.arange(steps) * dt
g      = np.array(hist['g'])
g_pred = np.array(hist['g_pred'])

# stable flip detection inside ramp window
stable_n = max(1, int(STABLE_S / dt))
start_idx = int((RAMP_START - 10.0) / dt)
end_idx   = int((RAMP_END + 5.0) / dt)

pred_idx = first_stable_index(g_pred, 1, stable_n, start_idx, end_idx)
react_idx= first_stable_index(g,      1, stable_n, start_idx, end_idx)
lead_time = (react_idx - pred_idx) * dt if (pred_idx is not None and react_idx is not None and pred_idx < react_idx) else np.nan

# windowed potential lead: compare Vhat(t) with V(t+H)
shift = int(round(PRED_HORIZON_S / dt))
mask  = (tvec >= RAMP_START) & (tvec <= RAMP_END)
V     = np.array(hist['V'])
Vhat  = np.array(hist['Vhat'])
if shift > 0:
    V_shift = np.roll(V, -shift)  # V(t+H)
    # keep only valid window
    Vw     = V_shift[mask]
    Vhatw  = Vhat[mask]
    mV     = Vw - Vw.mean()
    mVhat  = Vhatw - Vhatw.mean()
    # correlation as quality of lead fit
    denom = (np.linalg.norm(mV)*np.linalg.norm(mVhat) + 1e-9)
    corr_forward = float(np.dot(mV, mVhat) / denom)
else:
    corr_forward = np.nan

print(f"\nStable MPC Channel Lead Time: {np.nan if np.isnan(lead_time) else round(lead_time,3)} s")
print(f"Corr[ V̂(t), V(t+H) ] in {int(RAMP_START)}–{int(RAMP_END)} s window: {corr_forward:.3f}")

# ====================== 7) PLOTS ======================
Ehist = np.stack(hist['E'])
fig = plt.figure(figsize=(12, 11))
for tt in [50,200,240,int(RAMP_START),int(RAMP_END)]:
    plt.axvline(tt, color='k', ls='--', alpha=0.25)

# Potential
ax1 = plt.subplot(5,1,1)
ax1.plot(tvec, hist['V'], label=r'$V(t)$')
ax1.plot(tvec, hist['Vhat'], '--', label=r'$\hat V(t)$')
ax1.set_ylabel('Potential'); ax1.legend()

# Energy
ax2 = plt.subplot(5,1,2)
ax2.plot(tvec, Ehist[:,0], label=r'$E_p$')
ax2.plot(tvec, Ehist[:,1], label=r'$E_c$')
ax2.axhline(theta_p_on, color='gray',   ls=':')
ax2.axhline(theta_c_on, color='orange', ls=':')
ax2.set_ylabel('Energy'); ax2.legend()

# Channels
ax3 = plt.subplot(5,1,3)
ax3.plot(tvec, g, 'b', lw=2, drawstyle='steps-post', label='g')
ax3.plot(tvec, g_pred, 'r--', lw=2, drawstyle='steps-post', label='g_pred')
ax3.set_yticks([0,1,2]); ax3.set_yticklabels(['phys','cog','rel'])
ax3.set_ylabel('Channel'); ax3.legend()

# Uncertainty / boost
ax4 = plt.subplot(5,1,4)
ax4.plot(tvec, hist['traceSigma'], label=r'tr$(\Sigma)$')
ax4_t = ax4.twinx()
ax4_t.plot(tvec, hist['boost_phi'], 'm--', alpha=0.7, label=r'$\Phi$ boost')
ax4.set_ylabel('Uncertainty'); ax4_t.set_ylabel('Boost', color='m')
ax4.legend(loc='upper left'); ax4_t.legend(loc='upper right')

# Orientation
ax5 = plt.subplot(5,1,5)
Phi_x = [p[0] for p in hist['Phi']]
ax5.plot(tvec, Phi_x, label=r'$\Phi_x$')
ax5.set_xlabel('time (s)'); ax5.set_ylabel(r'$\Phi_x$'); ax5.legend()

plt.tight_layout()
plt.show()
