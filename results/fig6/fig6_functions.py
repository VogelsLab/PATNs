import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from scipy.signal import hilbert
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


# =====================================================================
# Gain-modulated simulation
# =====================================================================
def solve_with_gain(init_condition, W, tauE, tauI, T, dt, a, b, c, bias, gain):
    """
    Simulate with row-wise gain modulation: W_eff[i, :] *= gain[i].

    Parameters
    ----------
    gain : np.ndarray, shape (n,) — multiplicative gain per row (first n rows)
    """
    from utils import solve_rk4_springBox
    n = len(gain)
    W_eff = W.copy()
    W_eff[:n, :] *= gain[:, None]
    return solve_rk4_springBox(init_condition, W_eff, tauE, tauI, T, dt, a, b, c, bias)


# =====================================================================
# Frequency estimation
# =====================================================================
def estimate_freq_fft(t, y, burn_in=0.0):
    """Dominant frequency via FFT power spectrum."""
    mask = t >= burn_in
    t, y = t[mask], y[mask]
    y = y - y.mean()
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=t[1] - t[0])
    power = np.abs(Y)**2
    power[0] = 0.0
    return float(freqs[np.argmax(power)])


def estimate_freq_hilbert(t, y, burn_in=0.0):
    """
    Dominant frequency via Hilbert phase slope.

    Parameters
    ----------
    t, y     : time vector and signal (one neuron)
    burn_in  : time to discard before estimating

    Returns
    -------
    f : float — estimated frequency
    """
    mask = t >= burn_in
    t, y = t[mask], y[mask]
    y = y - np.mean(y)
    phase = np.unwrap(np.angle(hilbert(y)))
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, phase, rcond=None)[0]
    return float(slope / (2 * np.pi))


# =====================================================================
# Frequency gradient and gain direction
# =====================================================================
def freq_for_gains(g, W, tauE, tauI, T, dt, a, b, c, bias,
                   init_condition, burn_in, neuron_index_for_fft):
    """
    Apply row-wise gains g to W, simulate, return dominant frequency for one neuron.

    Parameters
    ----------
    g                    : np.ndarray, shape (n,) — gains for first n rows
    neuron_index_for_fft : int — which neuron to use for frequency estimation

    Returns
    -------
    f_dom : float
    r     : np.ndarray, shape (T/dt+1, N) — trajectory
    """
    from utils import solve_rk4_springBox
    n = len(g)
    W_eff = W.copy()
    W_eff[:n, :] *= g[:, None]
    t, r = solve_rk4_springBox(init_condition, W_eff, tauE, tauI, T, dt, a, b, c, bias)
    return estimate_freq_hilbert(t, r[:, neuron_index_for_fft], burn_in=burn_in), r


def compute_v(W, tauE, tauI, T, dt, a, b, c, bias,
              init_condition, burn_in, neuron_index_for_fft,
              n=100, eps=0.05, g_base=None):
    """
    Compute the normalized frequency gradient v = grad_g f / ||grad_g f||
    at baseline gains using central differences.

    Parameters
    ----------
    n      : int — number of gain dimensions (first n rows of W)
    eps    : float — finite-difference step
    g_base : np.ndarray or None — baseline gains (defaults to ones)

    Returns
    -------
    v    : np.ndarray, shape (n,) — normalized gradient
    f0   : float — baseline frequency
    grad : np.ndarray, shape (n,) — raw gradient
    """
    g0 = np.ones(n) if g_base is None else np.array(g_base, float).copy()
    grad = np.zeros(n)

    f0, _ = freq_for_gains(g0, W, tauE, tauI, T, dt, a, b, c, bias,
                            init_condition, burn_in, neuron_index_for_fft)

    for j in range(n):
        g_plus, g_minus = g0.copy(), g0.copy()
        g_plus[j]  += eps
        g_minus[j] -= eps
        fp, _ = freq_for_gains(g_plus,  W, tauE, tauI, T, dt, a, b, c, bias,
                                init_condition, burn_in, neuron_index_for_fft)
        fm, _ = freq_for_gains(g_minus, W, tauE, tauI, T, dt, a, b, c, bias,
                                init_condition, burn_in, neuron_index_for_fft)
        grad[j] = (fp - fm) / (2.0 * eps)

    v = grad.copy()
    norm_v = np.linalg.norm(v)
    if norm_v > 0:
        v /= norm_v
    return v, f0, grad


# =====================================================================
# Gain-modulated RK4 solver with real-time alpha schedule
# =====================================================================
def solve_rk4_springBox_gainmod(init_condition, W, tauE, tauI, T, dt,
                                 a, b, c, bias, v, alphas, alpha_fun, n=100):
    """
    RK4 solver with real-time gain modulation along direction v.

    At each time step t, gain g = 1 + alpha_fun(t) * v is applied to the
    first n rows of W.

    Parameters
    ----------
    v         : np.ndarray, shape (n,) — gain direction (normalized gradient)
    alphas    : list of float — alpha values passed to alpha_fun
    alpha_fun : callable(t, freqs_alpha, alphas) -> float
    n         : int — number of rows to modulate

    Returns
    -------
    tvec : np.ndarray
    r    : np.ndarray, shape (T/dt+1, N)
    """
    from utils import rk4_step
    tvec = np.arange(0, T + dt, dt)
    N = len(init_condition)
    r = np.zeros((N, len(tvec)))
    r[:, 0] = init_condition
    external_input = np.zeros(W.shape[0])
    W_base = W.copy()

    for i in range(1, len(tvec)):
        alpha = alpha_fun(tvec[i - 1], alphas)
        g = 1.0 + alpha * v
        W_eff = W_base.copy()
        W_eff[:n, :] *= g[:, None]
        r[:, i] = rk4_step(r[:, i-1], W_eff, external_input, tauE, tauI, dt, a, b, c, bias)

    return tvec, r.T


# =====================================================================
# Alpha schedule and event construction
# =====================================================================
def alpha_schedule(t, freqs_alpha, alphas, burn_in=2.0, alpha_burn=0.0):
    """
    Piecewise-constant alpha(t): burn-in period, then one segment per alpha,
    each lasting 1/f seconds.

    Parameters
    ----------
    freqs_alpha : list of float — frequencies for each alpha
    alphas      : list of float — alpha values
    burn_in     : float — duration of initial burn-in period
    alpha_burn  : float — alpha during burn-in
    """
    if t < burn_in:
        return alpha_burn
    t_shift = t - burn_in
    cumulative = 0.0
    for alpha, f in zip(alphas, freqs_alpha):
        cumulative += 1.0 / f
        if t_shift < cumulative:
            return alpha
    return 0.0


def build_input_events(freqs_alpha, burn_in=2.0, include_burn_event=True):
    """
    Build list of (t_start, t_end) intervals: optional burn-in + one per alpha.

    Parameters
    ----------
    freqs_alpha         : list of float
    burn_in             : float
    include_burn_event  : bool

    Returns
    -------
    events : list of (float, float)
    """
    events = []
    t_start = 0.0
    if include_burn_event and burn_in > 0:
        events.append((t_start, burn_in))
        t_start = burn_in
    for f in freqs_alpha:
        t_end = t_start + 1.0 / f
        events.append((t_start, t_end))
        t_start = t_end
    return events


# =====================================================================
# PCA and speed-coding analysis (notebook d)
# =====================================================================
def ls_solution(X, v):
    """Least-squares solution d s.t. Xd ≈ v, normalized to unit norm."""
    d_ls, *_ = np.linalg.lstsq(X, v, rcond=None)
    return d_ls / np.linalg.norm(d_ls), d_ls


def constrained_unit_norm_solution(X, v, tol=1e-9):
    """
    Solve min ||Xd - v|| subject to ||d|| = 1 via Lagrange multiplier bisection.
    """
    XtX = X.T @ X
    Xtv = X.T @ v
    eigvals = np.linalg.eigvalsh(XtX)
    lam_low  = -eigvals[-1] + 1e-9
    lam_high = 1e6

    def d_lam(lam):
        return np.linalg.solve(XtX + lam * np.eye(X.shape[1]), Xtv)

    for _ in range(100):
        lam_mid = 0.5 * (lam_low + lam_high)
        if np.dot(d_lam(lam_mid), d_lam(lam_mid)) > 1.0:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    d = d_lam(lam_mid)
    return d / np.linalg.norm(d)


def compare_methods(X, v):
    """Compare LS and constrained unit-norm solutions. Prints diagnostics."""
    d_norm_ls, d_ls = ls_solution(X, v)
    d_con = constrained_unit_norm_solution(X, v)
    cos = np.clip(np.dot(d_norm_ls, d_con), -1, 1)
    print(f'‣ ||d_ls|| = {np.linalg.norm(d_ls):.4g}')
    print(f'‣ Angle between directions [deg]: {np.degrees(np.arccos(cos)):.2f}')
    print(f'‣ ||Xd_ls - Xd_constrained|| = {np.linalg.norm(X @ d_norm_ls - X @ d_con):.4g}')
    return d_norm_ls, d_con


# =====================================================================
# Limit-cycle landing + phase delays (notebook e)
# =====================================================================
def estimate_period_steps(y, min_period=20, max_period=None):
    """Estimate dominant period in samples via FFT."""
    y = np.asarray(y)
    T = y.shape[0]
    if max_period is None:
        max_period = max(2, T // 2)
    y0 = y - np.mean(y)
    Y = np.fft.rfft(y0)
    freqs = np.fft.rfftfreq(T, d=1.0)
    power = Y.real**2 + Y.imag**2
    power[0] = 0.0
    f_min, f_max = 1.0 / min_period, 1.0 / max_period
    valid = (freqs >= f_max) & (freqs <= f_min)
    idx = np.where(valid)[0][np.argmax(power[valid])] if np.any(valid) else np.argmax(power[1:]) + 1
    f_dom = freqs[idx]
    period_steps = int(round(1.0 / f_dom)) if f_dom > 0 else min_period
    return max(min_period, min(period_steps, max_period))


def extract_reference_cycle(r_base, frac_discard=0.5, n_periods=1,
                             neuron_for_period=0, min_period=20):
    """
    Extract a reference cycle template from the tail of r_base.

    Returns
    -------
    r_cyc           : np.ndarray, shape (L, N)
    period_steps    : int
    idx_cycle_start : int
    """
    r_base = np.asarray(r_base)
    T, N = r_base.shape
    start_est = int(T * frac_discard)
    r_tail = r_base[start_est:, :]
    period_steps = estimate_period_steps(
        r_tail[:, neuron_for_period],
        min_period=min_period,
        max_period=max(30, r_tail.shape[0] // 2)
    )
    L = min(n_periods * period_steps, T)
    idx_cycle_start = T - L
    return r_base[idx_cycle_start:, :], period_steps, idx_cycle_start


def fit_pca_on_cycle(r_cyc, n_components=2):
    """Fit PCA on reference cycle. Returns (pca, X_cyc)."""
    pca = PCA(n_components=n_components)
    return pca, pca.fit_transform(np.asarray(r_cyc))


def _unwrap_cyclic_indices(idxs, L):
    idxs = np.asarray(idxs, dtype=float)
    out = idxs.copy()
    for k in range(1, len(out)):
        delta = out[k] - out[k-1]
        delta_wrapped = (delta + L/2) % L - L/2
        out[k] = out[k-1] + delta_wrapped
    return out


def find_landing_100D(r_traj, tree, r_cyc, period_steps,
                      eps_factor=0.02, min_window_periods=1.0,
                      min_progress_cycles=0.8):
    """
    Detect when r_traj enters and stays within a tube around r_cyc (100D).

    Returns
    -------
    j_land, i_land, d_min, i_star, eps
    """
    r_traj = np.asarray(r_traj)
    r_cyc  = np.asarray(r_cyc)
    L, T   = r_cyc.shape[0], r_traj.shape[0]

    center = r_cyc.mean(axis=0)
    R_typ  = np.median(np.linalg.norm(r_cyc - center[None, :], axis=1))
    eps    = eps_factor * R_typ

    d_min, i_star = tree.query(r_traj, k=1)
    window_len = max(2, min(int(round(min_window_periods * period_steps)), T))

    for j in range(T - window_len + 1):
        if np.max(d_min[j:j + window_len]) >= eps:
            continue
        win_u = _unwrap_cyclic_indices(i_star[j:j + window_len], L)
        if win_u[-1] - win_u[0] < min_progress_cycles * L:
            continue
        return j, int(i_star[j]), d_min, i_star, eps

    return None, None, d_min, i_star, eps


def landing_and_relative_phase(r_base, rs_ic, r_cyc, period_steps,
                                base_search_end=None, eps_factor=0.02,
                                min_window_periods=1.0, min_progress_cycles=0.8):
    """
    Compute landing + relative phase delay for all trajectories in rs_ic.

    Phase is theta = 2π i_land / L; relative phase = theta_k - theta_base
    wrapped to [-π, π].

    Returns
    -------
    res : dict with keys 'r_cyc', 'period_steps', 'base', 'trajectories'
    """
    r_cyc  = np.asarray(r_cyc)
    L      = r_cyc.shape[0]
    tree   = cKDTree(r_cyc)

    r_base_search = np.asarray(r_base) if base_search_end is None else np.asarray(r_base)[:base_search_end]
    base_j, base_i, base_d, base_istar, base_eps = find_landing_100D(
        r_base_search, tree, r_cyc, period_steps,
        eps_factor=eps_factor, min_window_periods=min_window_periods,
        min_progress_cycles=min_progress_cycles)
    base_theta = None if base_i is None else 2 * np.pi * base_i / float(L)

    infos = []
    for r_traj in rs_ic:
        j, i, d_min, i_star, eps = find_landing_100D(
            r_traj, tree, r_cyc, period_steps,
            eps_factor=eps_factor, min_window_periods=min_window_periods,
            min_progress_cycles=min_progress_cycles)
        theta = None if i is None else 2 * np.pi * i / float(L)
        if theta is not None and base_theta is not None:
            theta_rel_signed = float(((theta - base_theta) + np.pi) % (2*np.pi) - np.pi)
            theta_rel_2pi    = float((theta - base_theta) % (2*np.pi))
        else:
            theta_rel_signed = theta_rel_2pi = None
        infos.append(dict(j_land=j, i_land=i, theta_land=theta,
                          theta_rel_signed=theta_rel_signed,
                          theta_rel_2pi=theta_rel_2pi,
                          d_min=d_min, i_star=i_star, eps=eps))

    return dict(r_cyc=r_cyc, period_steps=period_steps,
                base=dict(j_land=base_j, i_land=base_i,
                          theta_land=base_theta, eps=base_eps),
                trajectories=infos)


def make_circle_ics_from_ic1(ic_orbit1, pca, X_base_cycle,
                              n_ic=12, radius_frac=0.03,
                              phase0=0.0, keep_ic1=True):
    """
    Create ICs by drawing a small circle in PC1-PC2 plane around ic_orbit1.

    Returns
    -------
    ics   : list of np.ndarray, shape (N,)
    X_ics : np.ndarray, shape (K, 2) — PC coordinates
    """
    ic_orbit1 = np.asarray(ic_orbit1)
    X0 = pca.transform(ic_orbit1[None, :])[0]
    center = X_base_cycle.mean(axis=0)
    R_typical = np.median(np.linalg.norm(X_base_cycle - center, axis=1))
    r_ic = radius_frac * R_typical
    angles = phase0 + np.linspace(0, 2*np.pi, n_ic, endpoint=False)
    X_ics = X0[None, :] + r_ic * np.c_[np.cos(angles), np.sin(angles)]
    ics = [pca.inverse_transform(X_ics[k:k+1, :])[0] for k in range(n_ic)]
    if keep_ic1:
        ics   = [ic_orbit1] + ics
        X_ics = np.vstack([X0[None, :], X_ics])
    return ics, X_ics


def make_ray_ics_from_ic1(ic_orbit1, pca, X_base_cycle,
                           n_ic=12, distances_frac=None,
                           angle=None, phase0=0.0,
                           keep_ic1=True, both_sides=False):
    """
    Create ICs along a ray in the PC1-PC2 plane from ic_orbit1.

    Parameters
    ----------
    angle          : float or None — direction in radians; defaults to phase0
    distances_frac : array-like or None — fractions of orbit radius
    both_sides     : bool — place points on both ± sides

    Returns
    -------
    ics   : list of np.ndarray
    X_ics : np.ndarray, shape (K, 2)
    """
    ic_orbit1 = np.asarray(ic_orbit1)
    X0 = pca.transform(ic_orbit1[None, :])[0]
    center = X_base_cycle.mean(axis=0)
    R_typical = np.median(np.linalg.norm(X_base_cycle - center, axis=1))
    u = np.array([np.cos(angle if angle is not None else phase0),
                  np.sin(angle if angle is not None else phase0)], dtype=float)

    if distances_frac is None:
        distances_frac = np.linspace(0.01, 0.12, n_ic)
    distances_frac = np.asarray(distances_frac, dtype=float)

    if both_sides:
        d = distances_frac * R_typical
        X_ics = X0[None, :] + d[:, None] * u[None, :]
    else:
        if distances_frac.size != n_ic:
            raise ValueError('distances_frac must have length n_ic when both_sides=False.')
        d = distances_frac * R_typical
        X_ics = X0[None, :] + d[:, None] * u[None, :]

    ics = [pca.inverse_transform(X_ics[k:k+1, :])[0] for k in range(X_ics.shape[0])]
    if keep_ic1:
        ics   = [ic_orbit1] + ics
        X_ics = np.vstack([X0[None, :], X_ics])
    return ics, X_ics


# =====================================================================
# Plotting helpers
# =====================================================================
def discrete_shades(cmap_name, n, light=0.30, dark=0.95):
    """Return n RGBA colors sampled from cmap_name, light -> dark."""
    return cm.get_cmap(cmap_name)(np.linspace(light, dark, n))


def extract_one_cycle_valley(t, r_alpha, i_neuron, f_alpha,
                              burn_in_plot, dt, n_cycles=1):
    """
    Extract one cycle of neuron i_neuron from r_alpha, aligned to a valley
    after burn_in_plot.

    Returns
    -------
    t_seg : np.ndarray — time starting at 0
    y_seg : np.ndarray — signal values
    """
    T_alpha  = 1.0 / f_alpha
    start_idx = np.searchsorted(t, burn_in_plot)
    search_end = min(start_idx + int(1.5 * T_alpha / dt), len(t))
    valley_rel = np.argmin(r_alpha[start_idx:search_end, i_neuron])
    anchor_idx = start_idx + valley_rel
    end_idx = min(anchor_idx + int(n_cycles * T_alpha / dt), len(t))
    t_seg = t[anchor_idx:end_idx] - t[anchor_idx]
    y_seg = r_alpha[anchor_idx:end_idx, i_neuron]
    return t_seg, y_seg


def plot_onecycle_overlay_for_neuron(i_neuron, colors, *,
                                      freqs_alpha, rs_alpha, t,
                                      burn_in_plot=4.0, dt=0.01,
                                      n_cycles=1, offset=0,
                                      figsize=(1.0, 0.7), dpi=500, lw=0.5):
    """
    Overlay one cycle per alpha for a single neuron, colored by alpha magnitude.

    Parameters
    ----------
    i_neuron : int — neuron index to plot
    colors   : array-like of RGBA — one color per alpha
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for k, (f_alpha, r_alpha, color) in enumerate(zip(freqs_alpha, rs_alpha, colors)):
        t_seg, y_seg = extract_one_cycle_valley(
            t, r_alpha, i_neuron, f_alpha, burn_in_plot, dt, n_cycles=n_cycles)
        ax.plot(t_seg, y_seg + k * offset, color=color, linewidth=lw)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.1, 1.3); ax.set_ylim(-0.4, 0.4)
    plt.tight_layout()
    return fig, ax


def plot_interval_gradient(ax, t, y, bounds, cmap_name, lw=0.6,
                            alpha=0.85, dark=0.95, light=0.35, zorder=3):
    """
    Plot y(t) with color changing per interval in bounds (dark -> light).

    Parameters
    ----------
    bounds    : list of float — interval boundaries including start and end
    cmap_name : str — matplotlib colormap name
    """
    t, y = np.asarray(t), np.asarray(y)
    pts  = np.column_stack([t, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    t_mid = 0.5 * (t[:-1] + t[1:])
    interval_idx = np.digitize(t_mid, bounds[1:-1], right=False)
    n_intervals  = len(bounds) - 1
    ramp  = np.linspace(dark, light, n_intervals)
    cvals = ramp[interval_idx]
    colors = cm.get_cmap(cmap_name)(cvals)
    lc = LineCollection(segs, colors=colors, linewidths=lw, alpha=alpha, zorder=zorder)
    ax.add_collection(lc)
    return lc
