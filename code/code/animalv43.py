# animalv43.py
import numpy as np
import matplotlib.pyplot as plt

# ====================== PARAMETERS ======================
alpha_f = 2.0;   alpha_E = 1.5;   alpha_phi = 3.5
beta_f  = 0.1;   beta = np.array([0.3, 0.2, 0.1])
Gamma   = np.array([0.8, 0.6, 0.4])
eta_E   = 0.05

w_f = 1.0
W_E = np.diag([1.2, 0.8, 0.6])
lambda_phi = 3.0

f0 = 1.0
E0 = np.array([1.0, 0.7, 0.5])
theta_p_on, theta_p_off = 0.62, 0.58
theta_c_on, theta_c_off = 0.42, 0.38

tau_m = {'f':2.0, 'Ep':1.5, 'Ec':2.0, 'Er':3.0, 'Phi':1.0}
sigma_base = {'f':0.03, 'Ep':0.04, 'Ec':0.03, 'Er':0.02, 'Phi':0.05}

dt = 0.02
T  = 400
steps = int(T/dt)
rng = np.random.default_rng(42)

# ====================== STATE INIT ======================
f   = f0
E   = E0.copy()
Phi = np.array([1.0, 0.0, 0.0])
xi  = {k:0.0 for k in tau_m}
gate_p_open = True
gate_c_open = False

hist = {k:[] for k in ['f','E','Phi','V','Vhat','g','g_pred','I','f_env','s','traceSigma','boost_phi']}

# ====================== HELPERS ======================
def policy_dir(g):
    dirs = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    return np.array(dirs[g])

def project_tangent(v, phi):
    return v - np.dot(v, phi) * phi

# ====================== LONG-HORIZON LOOKAHEAD ======================
def predict_state_ahead(E_cur, f_cur, shat, horizon_s, dt_outer, params, gate_state):
    alpha_f, alpha_E, beta, Gamma, W_E, w_f, f0, E0 = params
    dt_h = dt_outer / 5.0
    steps = max(1, int(horizon_s / dt_h))
    Eh, fh = E_cur.copy(), f_cur
    p_open, c_open = gate_state['p'], gate_state['c']

    for _ in range(steps):
        # Hysteresis
        if p_open and Eh[0] < theta_p_off: p_open = False
        elif not p_open and Eh[0] > theta_p_on: p_open = True
        if p_open and c_open and Eh[1] < theta_c_off: c_open = False
        elif p_open and not c_open and Eh[1] > theta_c_on: c_open = True
        I_pred = np.array([1.0, 1.0 if c_open else 0.0, 0.0])

        # Predicted channel
        e_hat = np.maximum(E0 - Eh, 0.0) + 0.2 * shat
        g_hat = int(np.argmax(e_hat))
        u_hat = np.eye(3)[g_hat]

        # Drifts
        df_det = -2*alpha_f*w_f*(fh - f0)
        dE_det = -2*alpha_E*W_E @ (Eh - E0)
        dE_rec = beta * I_pred * (E0 - Eh)
        dE_eff = -Gamma * u_hat

        fh += df_det * dt_h
        Eh += (dE_det + dE_rec + dE_eff) * dt_h
        Eh = np.clip(Eh, 0.0, None)

    return Eh, fh, p_open, c_open

# ====================== CIL CLASS ======================
class CIL:
    def __init__(self, dt, d=4, alpha_mpc=4.5, curiosity_gamma=1.6, ema_rho=0.002):
        self.dt = dt; self.d = d
        self.M = np.zeros(d); self.S = np.eye(d) * 0.2
        self.A = np.eye(d) * 0.98
        self.C = np.eye(d)
        self.Q = np.eye(d) * 5e-3
        self.R = np.eye(d) * 8e-3
        self.alpha_mpc = alpha_mpc
        self.curiosity_gamma = curiosity_gamma
        self.ema_rho = ema_rho
        self.prev_obs = None

        self.A += np.array([
            [0.00, 0.03, 0.00, 0.00],
            [0.02, 0.00, 0.04, 0.00],
            [0.00, 0.03, 0.00, 0.02],
            [0.00, 0.00, 0.02, 0.00],
        ])
        self.A *= 0.98

    def predict(self):  M_ = self.A @ self.M; S_ = self.A @ self.S @ self.A.T + self.Q; return M_, S_
    def forecast(self): return self.A @ self.M

    def correct(self, obs):
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
        return self.M

cil = CIL(dt, alpha_mpc=4.5, curiosity_gamma=1.6, ema_rho=0.002)

# ====================== MAIN LOOP ======================
for step in range(steps):
    t = step * dt

    # ---- Environment (ramp starts early) ----
    base_osc = 0.10 * np.sin(0.25 * t)
    if t < 50:
        f_env = f0 + base_osc; s = np.zeros(3)
    elif t < 200:
        f_env = f0 + base_osc; s = np.array([1.5, 0.0, 0.0])
    elif t < 240:
        f_env = f0 + 0.8 * np.sin(12 * t); s = np.array([0.3, 0.0, 0.0])
    else:
        f_env = f0 + base_osc
        ramp = np.clip((t - 255) / 45, 0.0, 1.0)  # starts earlier, smoother
        s = np.array([0.0, 1.2 * ramp, 0.0])
        if (t - 240) % 15 < dt * 2 and rng.random() < 0.7:
            xi['Phi'] += 0.45

    # ---- CIL update ----
    obs = np.concatenate([[f_env], s])
    cil.correct(obs)
    fhat, sp, sc, sr = cil.forecast()
    shat = np.array([sp, sc, sr])

    # ---- LONG-HORIZON LOOKAHEAD (2.0 s) ----
    params = (alpha_f, alpha_E, beta, Gamma, W_E, w_f, f0, E0)
    gate_init = {'p': gate_p_open, 'c': gate_c_open}
    E_hat_pred, f_hat_pred, _, _ = predict_state_ahead(
        E, f, shat, horizon_s=2.0, dt_outer=dt, params=params, gate_state=gate_init
    )

    # ---- PREDICTED CHANNEL (future E) ----
    e_hat = np.maximum(E0 - E_hat_pred, 0.0) + 0.2 * shat
    # Tie-break toward cognitive
    if abs(e_hat[1] - e_hat[0]) <= 0.01 * (1.0 + e_hat.max()):
        e_hat[1] += 1e-6
    g_pred = int(np.argmax(e_hat))
    Phi_star_pred = policy_dir(g_pred)

    # ---- CURRENT hysteresis gating ----
    if gate_p_open and E[0] < theta_p_off: gate_p_open = False
    elif not gate_p_open and E[0] > theta_p_on: gate_p_open = True
    if gate_p_open and gate_c_open and E[1] < theta_c_off: gate_c_open = False
    elif gate_p_open and not gate_c_open and E[1] > theta_c_on: gate_c_open = True
    I = np.array([1.0, 1.0 if gate_c_open else 0.0, 0.0])

    # ---- Reactive channel ----
    err = np.maximum(E0 - E, 0.0) + 0.2 * s
    g = int(np.argmax(err))
    u_g = np.eye(3)[g]
    Phi_star = policy_dir(g)

    # ---- Curiosity boost ----
    Sdiag = np.clip(np.diag(cil.S), 0, None)
    phi_boost = 1.0 + 1.2 * (0.5 * Sdiag[1] + 0.5 * Sdiag[2])
    sigma = {
        'f':   sigma_base['f']   * (1.0 + cil.curiosity_gamma * Sdiag[0]),
        'Ep':  sigma_base['Ep']  * (1.0 + cil.curiosity_gamma * Sdiag[1]),
        'Ec':  sigma_base['Ec']  * (1.0 + cil.curiosity_gamma * Sdiag[2]),
        'Er':  sigma_base['Er']  * (1.0 + cil.curiosity_gamma * Sdiag[3]),
        'Phi': sigma_base['Phi'] * phi_boost
    }

    # ---- OU update ----
    for k in xi:
        tau, sig = tau_m[k], sigma[k]
        xi[k] += (-xi[k]/tau) * dt + sig * np.sqrt(2/tau) * np.sqrt(dt) * rng.normal()

    # ---- Dynamics ----
    df_det = -2 * alpha_f * w_f * (f - f_env)
    dE_det = -2 * alpha_E * W_E @ (E - E0)
    dE_rec = beta * I * (E0 - E)
    dE_eff = -Gamma * u_g

    mis = 1.0 - np.dot(Phi, Phi_star)
    dPhi_det = -alpha_phi * project_tangent(2 * lambda_phi * mis * Phi_star, Phi)
    mis_pred = 1.0 - np.dot(Phi, Phi_star_pred)
    dPhi_mpc = -cil.alpha_mpc * project_tangent(2 * lambda_phi * mis_pred * Phi_star_pred, Phi)

    df = (df_det + beta_f * (f0 - f) + eta_E * np.sum(E) + xi['f']) * dt
    dE = (dE_det + dE_rec + dE_eff) * dt + np.array([xi['Ep'], xi['Ec'], xi['Er']]) * dt
    dPhi = (dPhi_det + dPhi_mpc + xi['Phi'] * rng.normal(size=3)) * dt

    f += df
    E += dE; E = np.clip(E, 0.0, None)
    Phi += dPhi; Phi /= np.linalg.norm(Phi)

    # ---- Diagnostics ----
    V = w_f * (f - f_env)**2 + (E - E0) @ W_E @ (E - E0) + lambda_phi * (1 - np.dot(Phi, Phi_star))**2
    Vhat = w_f * (f - f_hat_pred)**2 + (E - E0) @ W_E @ (E - E0) + lambda_phi * (1 - np.dot(Phi, Phi_star_pred))**2

    for k, v in zip(hist.keys(), [f,E,Phi,V,Vhat,g,g_pred,I,f_env,s,np.trace(cil.S),phi_boost]):
        hist[k].append(v if np.isscalar(v) else v.copy())

# ====================== METRICS ======================
tvec = np.arange(steps) * dt
g = np.array(hist['g'])
gpred = np.array(hist['g_pred'])
flip = np.diff(g, prepend=g[0]) != 0
flipp = np.diff(gpred, prepend=gpred[0]) != 0
lead_samples = [i - np.where(flipp[:i])[0][-1] for i in np.where(flip)[0] if np.any(flipp[:i])]
lead_time = np.mean(np.array(lead_samples) * dt) if lead_samples else np.nan

# Windowed V-lead (260–300s) — correct mean-centering
mask = (tvec >= 260) & (tvec <= 300)
V_arr = np.array(hist['V'])
Vhat_arr = np.array(hist['Vhat'])
Vw = V_arr[mask] - V_arr[mask].mean()
Vhatw = Vhat_arr[mask] - Vhat_arr[mask].mean()

def lead_seconds_centered(x, y, dt, max_lag_s=5.0):
    max_lag = int(max_lag_s/dt)
    best_k, best_c = 0, -1
    for k in range(-max_lag, max_lag+1):
        xc = np.roll(x, -k)
        c = np.dot(xc, y) / (np.linalg.norm(xc) * np.linalg.norm(y) + 1e-9)
        if c > best_c: best_c, best_k = c, k
    return best_k * dt

lead_V = lead_seconds_centered(Vhatw, Vw, dt)

print(f"\nMPC Channel Lead Time: {lead_time:.3f} s")
print(f"Lead of V̂ over V (260-300s): {lead_V:.3f} s")

# ====================== PLOTS ======================
Ehist = np.stack(hist['E'])
fig = plt.figure(figsize=(12,11))
for tt in [50,200,240,255,300]: plt.axvline(tt, color='k', ls='--', alpha=0.3)

plt.subplot(5,1,1); plt.plot(tvec, hist['V'], label=r'$V(t)$'); plt.plot(tvec, hist['Vhat'], '--', label=r'$\hat V(t)$'); plt.ylabel('Potential'); plt.legend()
plt.subplot(5,1,2); plt.plot(tvec, Ehist[:,0], label=r'$E_p$'); plt.plot(tvec, Ehist[:,1], label=r'$E_c$'); plt.axhline(theta_p_on, color='gray', ls=':'); plt.axhline(theta_c_on, color='orange', ls=':'); plt.ylabel('Energy'); plt.legend()
plt.subplot(5,1,3); plt.plot(tvec, hist['g'], 'b', lw=2, drawstyle='steps-post', label='g'); plt.plot(tvec, hist['g_pred'], 'r--', lw=2, drawstyle='steps-post', label='g_pred'); plt.yticks([0,1,2],['phys','cog','rel']); plt.ylabel('Channel'); plt.legend()
plt.subplot(5,1,4); plt.plot(tvec, hist['traceSigma'], label=r'tr$(\Sigma)$'); plt.twinx().plot(tvec, hist['boost_phi'], 'm--', alpha=0.7, label=r'$\Phi$ boost'); plt.ylabel('Uncertainty')
plt.subplot(5,1,5); plt.plot(tvec, [p[0] for p in hist['Phi']], label=r'$\Phi_x$'); plt.xlabel('time (s)'); plt.ylabel(r'$\Phi_x$'); plt.legend()

plt.tight_layout(); plt.show()