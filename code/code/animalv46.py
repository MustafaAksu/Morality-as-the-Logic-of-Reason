import numpy as np
import matplotlib.pyplot as plt

# ====================== CONFIG ======================
PRINT_DEBUG_FLIPS   = False
SEED                = 42

# Prediction horizon & weights
PRED_HORIZON_S      = 3.5      # seconds to look ahead (↑ for more lead)
PRED_KAPPA          = 0.20     # reactive stimulus weight
PRED_KAPPA_PRED     = 0.25     # predicted stimulus weight (slight bias helps earlier switch)
PRED_MARGIN         = 0.010    # min advantage (pred cog vs best alt) to accept predicted switch
STABLE_S            = 0.50     # seconds a flip must persist to be "stable"

# Ramp window (where we expect the switch)
RAMP_START, RAMP_END = 260.0, 300.0

# ====================== PARAMETERS ======================
alpha_f = 2.0;   alpha_E = 1.5;   alpha_phi = 3.5
beta_f  = 0.1;   beta = np.array([0.35, 0.20, 0.10])  # slight ↑ Ep recovery helps lead
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

# OU noise
tau_m = {'f':2.0, 'Ep':1.5, 'Ec':2.0, 'Er':3.0, 'Phi':1.0}
sigma_base = {'f':0.03, 'Ep':0.04, 'Ec':0.03, 'Er':0.02, 'Phi':0.05}

# Simulation clock
dt = 0.02
T  = 400
steps = int(T/dt)
rng = np.random.default_rng(SEED)

# ====================== STATE INIT ======================
f   = f0
E   = E0.copy()
Phi = np.array([1.0, 0.0, 0.0])
xi  = {k:0.0 for k in tau_m}

# Hysteresis states (actual)
gate_p_open = True
gate_c_open = False

# Storage
hist = {k:[] for k in [
    'f','E','Phi','V','Vhat','g','g_pred','I','f_env','s',
    'traceSigma','boost_phi','m','m_pred','c_pred','adv_hat'
]}

# ====================== HELPERS ======================
def policy_dir(g):
    dirs = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    return np.array(dirs[g])

def project_tangent(v, phi):
    return v - np.dot(v, phi) * phi

def first_stable_index(a, value, stable_samples, start_idx=0, end_idx=None):
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

# ---- long-horizon rollout (E,f) with predicted stimuli and gates ----
def predict_state_ahead(E_cur, f_cur, shat, horizon_s, dt_outer, params, gate_state):
    alpha_f, alpha_E, beta, Gamma, W_E, w_f, f0, E0 = params
    dt_h = dt_outer / 5.0
    steps = max(1, int(horizon_s / dt_h))
    Eh = E_cur.copy(); fh = f_cur
    p_open, c_open = gate_state['p'], gate_state['c']
    for _ in range(steps):
        # hysteresis on predicted E
        if p_open and Eh[0] < theta_p_off: p_open = False
        elif (not p_open) and Eh[0] > theta_p_on: p_open = True
        if p_open and c_open and Eh[1] < theta_c_off: c_open = False
        elif p_open and (not c_open) and Eh[1] > theta_c_on: c_open = True
        I_pred = np.array([1.0, 1.0 if c_open else 0.0, 0.0])
        # choose channel under predicted stimuli
        e_hat = np.maximum(E0 - Eh, 0.0) + PRED_KAPPA_PRED * shat
        if abs(e_hat[1] - e_hat[0]) <= 0.01 * (1.0 + e_hat.max()):
            e_hat[1] += 1e-6
        g_hat = int(np.argmax(e_hat))
        u_hat = np.eye(3)[g_hat]
        # deterministic drift (env held at f0 in short rollout)
        df_det = -2*alpha_f*w_f*(fh - f0)
        dE_det = -2*alpha_E*W_E @ (Eh - E0)
        dE_rec = beta * I_pred * (E0 - Eh)
        dE_eff = -Gamma * u_hat
        fh += df_det * dt_h
        Eh += (dE_det + dE_rec + dE_eff) * dt_h
        Eh = np.clip(Eh, 0.0, None)
    return Eh, fh, p_open, c_open

# ====================== CIL (predictive layer) ======================
class CIL:
    def __init__(self, dt, d=4, alpha_mpc=4.5, curiosity_gamma=1.6, ema_rho=0.002):
        self.dt = dt; self.d = d
        self.M = np.zeros(d); self.S = np.eye(d) * 0.2
        self.A = np.eye(d) * 0.98; self.C = np.eye(d)
        self.Q = np.eye(d) * 5e-3; self.R = np.eye(d) * 8e-3
        self.alpha_mpc = alpha_mpc; self.curiosity_gamma = curiosity_gamma
        self.ema_rho = ema_rho; self.prev_obs = None; self.prev_M = None
        # weak cross-coupling
        self.A += np.array([
            [0.00, 0.03, 0.00, 0.00],
            [0.02, 0.00, 0.04, 0.00],
            [0.00, 0.03, 0.00, 0.02],
            [0.00, 0.00, 0.02, 0.00],
        ]); self.A *= 0.98
    def predict(self):
        M_ = self.A @ self.M; S_ = self.A @ self.S @ self.A.T + self.Q; return M_, S_
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
        self.prev_obs = obs.copy(); self.prev_M = M_prev_for_vel
        return self.M
    def forecast_seconds(self, seconds, dt):
        if self.prev_M is None:  # fall back to one-step
            return self.A @ self.M
        v = (self.M - self.prev_M) / max(dt, 1e-9)
        return self.M + seconds * v

cil = CIL(dt, alpha_mpc=4.5, curiosity_gamma=1.6, ema_rho=0.002)

# ====================== MAIN LOOP ======================
for step in range(steps):
    t = step * dt

    # Environment
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

    # CIL update and future stimuli
    obs = np.concatenate([[f_env], s])
    cil.correct(obs)
    fhat, sp, sc, sr = cil.forecast_seconds(seconds=PRED_HORIZON_S, dt=dt)
    shat = np.array([sp, sc, sr])

    # Predict (E,f) and future gates over the horizon
    params = (alpha_f, alpha_E, beta, Gamma, W_E, w_f, f0, E0)
    gate_init = {'p': gate_p_open, 'c': gate_c_open}
    E_hat_pred, f_hat_pred, p_pred, c_pred = predict_state_ahead(
        E, f, shat, horizon_s=PRED_HORIZON_S, dt_outer=dt, params=params, gate_state=gate_init
    )

    # Predicted channel decision (must have future cog gate open and advantage > margin)
    e_hat = np.maximum(E0 - E_hat_pred, 0.0) + PRED_KAPPA_PRED * shat
    alt = max(e_hat[0], e_hat[2])
    advantage = e_hat[1] - alt
    if c_pred and (advantage > PRED_MARGIN): g_pred = 1
    else: g_pred = 0
    Phi_star_pred = policy_dir(g_pred)

    # Actual hysteresis gating (recovery)
    if gate_p_open and E[0] < theta_p_off: gate_p_open = False
    elif (not gate_p_open) and E[0] > theta_p_on: gate_p_open = True
    if gate_p_open and gate_c_open and E[1] < theta_c_off: gate_c_open = False
    elif gate_p_open and (not gate_c_open) and E[1] > theta_c_on: gate_c_open = True
    I = np.array([1.0, 1.0 if gate_c_open else 0.0, 0.0])

    # Reactive channel and orientation targets
    err = np.maximum(E0 - E, 0.0) + PRED_KAPPA * s
    g = int(np.argmax(err))
    u_g = np.eye(3)[g]
    Phi_star = policy_dir(g)

    # Curiosity / OU scaling
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

    # Deterministic dynamics
    df_det = -2 * alpha_f * w_f * (f - f_env)
    dE_det = -2 * alpha_E * W_E @ (E - E0)
    dE_rec = beta * I * (E0 - E)
    dE_eff = -Gamma * u_g

    mis = 1.0 - np.dot(Phi, Phi_star)        # reactive misalignment
    m_pred = 1.0 - np.dot(Phi, Phi_star_pred) # predicted misalignment

    dPhi_det = -alpha_phi * project_tangent(2 * lambda_phi * mis * Phi_star, Phi)
    dPhi_mpc = -cil.alpha_mpc * project_tangent(2 * lambda_phi * m_pred * Phi_star_pred, Phi)

    f   += (df_det + beta_f * (f0 - f) + eta_E * np.sum(E) + xi['f']) * dt
    E   += (dE_det + dE_rec + dE_eff) * dt + np.array([xi['Ep'], xi['Ec'], xi['Er']]) * dt
    E    = np.clip(E, 0.0, None)
    Phi += (dPhi_det + dPhi_mpc + xi['Phi'] * rng.normal(size=3)) * dt
    Phi /= np.linalg.norm(Phi)

    # Diagnostics (note: Vhat uses predicted energy)
    V = w_f * (f - f_env)**2 + (E - E0) @ W_E @ (E - E0) + lambda_phi * (1 - np.dot(Phi, Phi_star))**2
    Vhat = (
        w_f * (f - f_hat_pred)**2
        + (E_hat_pred - E0) @ W_E @ (E_hat_pred - E0)
        + lambda_phi * (1 - np.dot(Phi, Phi_star_pred))**2
    )

    # store
    for k, v in [
        ('f',f), ('E',E.copy()), ('Phi',Phi.copy()), ('V',V), ('Vhat',Vhat),
        ('g',g), ('g_pred',g_pred), ('I',I.copy()), ('f_env',f_env), ('s',s.copy()),
        ('traceSigma',np.trace(cil.S)), ('boost_phi',phi_boost),
        ('m',mis), ('m_pred',m_pred), ('c_pred',float(c_pred)), ('adv_hat',advantage)
    ]:
        hist[k].append(v)

# ====================== METRICS ======================
tvec = np.arange(steps)*dt
stable_n = max(1, int(STABLE_S/dt))
start_idx = int((RAMP_START - 10.0)/dt)
end_idx   = int((RAMP_END + 5.0)/dt)

g      = np.array(hist['g'])
g_pred = np.array(hist['g_pred'])
c_pred = np.array(hist['c_pred'])
adv    = np.array(hist['adv_hat'])

# (A) Stable channel lead inside the ramp window
pred_idx = first_stable_index(g_pred, 1, stable_n, start_idx, end_idx)
react_idx= first_stable_index(g,      1, stable_n, start_idx, end_idx)
lead_channel = (react_idx - pred_idx)*dt if (pred_idx is not None and react_idx is not None and pred_idx < react_idx) else np.nan

# (B) Gate lead: earliest step in window where predicted future gate is open AND advantage > margin (stable)
cond = (c_pred > 0.5) & (adv > PRED_MARGIN)
pred_gate_idx = first_stable_index(cond.astype(int), 1, stable_n, start_idx, end_idx)
react_gate_idx= first_stable_index((g==1).astype(int), 1, stable_n, start_idx, end_idx)
lead_gate = (react_gate_idx - pred_gate_idx)*dt if (pred_gate_idx is not None and react_gate_idx is not None and pred_gate_idx < react_gate_idx) else np.nan

# (C) Misalignment lead: when m_pred drops below 60% of its pre-ramp mean vs m
m      = np.array(hist['m']);      m_pred = np.array(hist['m_pred'])
pre_mask = (tvec >= (RAMP_START-20)) & (tvec < (RAMP_START-5))
thresh = 0.60 * max(1e-6, m_pred[pre_mask].mean())
t_pred_m = np.argmax((m_pred[start_idx:end_idx] < thresh).astype(int))
t_reac_m = np.argmax((m[start_idx:end_idx]     < thresh).astype(int))
lead_mis = (t_reac_m - t_pred_m)*dt if (t_pred_m>0 and t_reac_m>0 and t_pred_m < t_reac_m) else np.nan

# (D) Forward potential fit: corr( V̂(t), V(t+H) ) in window
shift = int(round(PRED_HORIZON_S/dt))
mask  = (tvec >= RAMP_START) & (tvec <= RAMP_END)
V     = np.array(hist['V']); Vhat = np.array(hist['Vhat'])
if shift > 0:
    Vw = V[mask]
    Vshift = np.roll(V, -shift)[mask]
    Vhatw  = Vhat[mask]
    mV = Vshift - Vshift.mean(); mVhat = Vhatw - Vhatw.mean()
    denom = (np.linalg.norm(mV)*np.linalg.norm(mVhat) + 1e-9)
    corr_forward = float(np.dot(mV, mVhat)/denom)
else:
    corr_forward = np.nan

print("\n--- LEAD METRICS ---")
print(f"Stable channel lead (ramp window): {np.nan if np.isnan(lead_channel) else round(lead_channel,3)} s")
print(f"Pred gate lead (future gate+adv>margin): {np.nan if np.isnan(lead_gate) else round(lead_gate,3)} s")
print(f"Misalignment lead (m_pred vs m): {np.nan if np.isnan(lead_mis) else round(lead_mis,3)} s")
print(f"Corr[ V̂(t), V(t+H) ] in {int(RAMP_START)}–{int(RAMP_END)} s: {corr_forward:.3f}")

# ====================== PLOTS ======================
Ehist = np.stack(hist['E'])
fig = plt.figure(figsize=(12,11))
for tt in [50,200,240,int(RAMP_START),int(RAMP_END)]: plt.axvline(tt, color='k', ls='--', alpha=0.25)

ax1 = plt.subplot(6,1,1)
ax1.plot(tvec, hist['V'], label=r'$V(t)$')
ax1.plot(tvec, hist['Vhat'], '--', label=r'$\hat V(t)$')
ax1.set_ylabel('Potential'); ax1.legend()

ax2 = plt.subplot(6,1,2)
ax2.plot(tvec, Ehist[:,0], label=r'$E_p$')
ax2.plot(tvec, Ehist[:,1], label=r'$E_c$')
ax2.axhline(theta_p_on, color='gray', ls=':'); ax2.axhline(theta_c_on, color='orange', ls=':')
ax2.set_ylabel('Energy'); ax2.legend()

ax3 = plt.subplot(6,1,3)
ax3.plot(tvec, g, 'b', lw=2, drawstyle='steps-post', label='g')
ax3.plot(tvec, g_pred, 'r--', lw=2, drawstyle='steps-post', label='g_pred')
ax3.set_yticks([0,1,2]); ax3.set_yticklabels(['phys','cog','rel'])
ax3.set_ylabel('Channel'); ax3.legend()

ax4 = plt.subplot(6,1,4)
ax4.plot(tvec, np.array(hist['traceSigma']), label=r'tr$(\Sigma)$')
ax4_t = ax4.twinx()
ax4_t.plot(tvec, np.array(hist['boost_phi']), 'm--', alpha=0.7, label=r'$\Phi$ boost')
ax4.set_ylabel('Uncertainty'); ax4_t.set_ylabel('Boost', color='m')
ax4.legend(loc='upper left'); ax4_t.legend(loc='upper right')

ax5 = plt.subplot(6,1,5)
ax5.plot(tvec, np.array(hist['m']),      label='m (reactive misalign)')
ax5.plot(tvec, np.array(hist['m_pred']), '--', label='m_pred (pred misalign)')
ax5.axhline(0.60 * max(1e-6, np.array(hist['m_pred'])[(tvec >= (RAMP_START-20)) & (tvec < (RAMP_START-5))].mean()),
            color='k', ls=':', alpha=0.5)
ax5.set_ylabel('misalignment'); ax5.legend()

ax6 = plt.subplot(6,1,6)
Phi_x = [p[0] for p in hist['Phi']]
ax6.plot(tvec, Phi_x, label=r'$\Phi_x$')
ax6.set_xlabel('time (s)'); ax6.set_ylabel(r'$\Phi_x$'); ax6.legend()

plt.tight_layout(); plt.show()
