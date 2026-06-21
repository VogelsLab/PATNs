import numpy as np
from scipy.special import expit


# =====================================================================
# Fixed point finding
# =====================================================================
def find_fixed_point(W, bias, r0, alpha=1e-3, max_iter=500_000,
                     tolF=1e-10, every=None):
    """
    Find a fixed point of the sigmoid network using gradient descent.
    Uses standard sigmoid (a=1, b=0, c=1) — i.e. f(x) = sigmoid(Wx + bias).

    Parameters
    ----------
    W       : np.ndarray, shape (N, N)
    bias    : np.ndarray, shape (N,)
    r0      : np.ndarray, shape (N,) — initial guess
    alpha   : float — step size
    tolF    : float — convergence tolerance on ||F(r)||
    every   : int or None — print diagnostic every this many iters

    Returns
    -------
    r      : np.ndarray, shape (N,) — fixed point estimate
    ok     : bool — True if converged
    info   : dict — 'iters', 'residual'
    """
    from utils import f as _f
    a = b = 0; c = 1
    a_arr = np.ones(len(r0)); b_arr = np.zeros(len(r0)); c_arr = np.ones(len(r0))

    r = np.clip(r0.copy(), -0.5, 0.5)
    for k in range(1, max_iter + 1):
        Fr    = -r + _f(W @ r + bias, a_arr, b_arr, c_arr)
        r_new = np.clip(r + alpha * Fr, -0.5, 0.5)
        if every is not None and k % every == 0:
            Fn   = np.linalg.norm(-r_new + _f(W @ r_new + bias, a_arr, b_arr, c_arr))
            step = np.linalg.norm(r_new - r)
            print(f'k={k}  step={step:.3e}  ||F||={Fn:.3e}')
        if np.linalg.norm(Fr) < tolF:
            return r, True, {'iters': k, 'residual': float(np.linalg.norm(Fr))}
        r = r_new
    resid = float(np.linalg.norm(-r + _f(W @ r + bias, a_arr, b_arr, c_arr)))
    return r, False, {'iters': max_iter, 'residual': resid}


def make_bias(N, beta, eps, rng, zero_mean=True):
    """
    Create bias vector b = beta*1 + eps*eta, where eta ~ Uniform(-1,1).

    Returns
    -------
    b   : np.ndarray, shape (N,)
    eta : np.ndarray, shape (N,)
    """
    eta = rng.uniform(-1, 1, N)
    if zero_mean:
        eta -= eta.mean()
    return beta * np.ones(N) + eps * eta, eta


def make_many_biases(N, betas, epsilons, seeds=0):
    """
    Generate K bias vectors for different (beta, eps) combinations.

    Parameters
    ----------
    betas, epsilons : scalar or array of length K
    seeds           : scalar seed or array of length K

    Returns
    -------
    biases : np.ndarray, shape (N, K)
    etas   : np.ndarray, shape (N, K)
    """
    betas    = np.atleast_1d(betas)
    epsilons = np.atleast_1d(epsilons)
    if betas.size    == 1: betas    = np.repeat(betas,    epsilons.size)
    if epsilons.size == 1: epsilons = np.repeat(epsilons, betas.size)
    K = len(betas)
    seeds = np.atleast_1d(seeds)
    if seeds.size == 1:
        seeds = np.arange(K) + int(seeds[0])

    biases = np.zeros((N, K))
    etas   = np.zeros((N, K))
    for k in range(K):
        rng_k = np.random.default_rng(seeds[k])
        biases[:, k], etas[:, k] = make_bias(N, betas[k], epsilons[k], rng_k)
    return biases, etas


def find_fixed_points(W, biases, r0s, alpha=1e-3, max_iter=500_000,
                      tolF=1e-10, every=None):
    """
    Find fixed points for each column of biases.

    Parameters
    ----------
    biases : np.ndarray, shape (N, K)
    r0s    : np.ndarray, shape (N, K) — initial guesses

    Returns
    -------
    r_fps : np.ndarray, shape (N, K)
    ok    : np.ndarray, shape (K,) bool
    info  : dict with 'iters' and 'residual' arrays of shape (K,)
    """
    N, K = biases.shape
    r_fps  = np.zeros((N, K), dtype=float)
    ok_arr = np.zeros(K, dtype=bool)
    iters  = np.zeros(K, dtype=int)
    resid  = np.zeros(K, dtype=float)

    for k in range(K):
        r_fp, ok, info = find_fixed_point(
            W, biases[:, k], r0s[:, k],
            alpha=alpha, max_iter=max_iter, tolF=tolF)
        r_fps[:, k]  = r_fp
        ok_arr[k]    = ok
        iters[k]     = info['iters']
        resid[k]     = info['residual']
        if every is not None and k % every == 0:
            print(f'k={k}/{K}  ok={ok}  iters={iters[k]}  resid={resid[k]:.3e}')

    return r_fps, ok_arr, {'iters': iters, 'residual': resid}


# =====================================================================
# Jacobian at a fixed point
# =====================================================================
def fprime(z):
    """Derivative of the standard sigmoid: f'(z) = sigmoid(z)*(1-sigmoid(z))."""
    s = expit(z)
    return s * (1 - s)


def jacobian_at_rstar(W, bias, r_star):
    """
    Jacobian of the sigmoid network at fixed point r*:
        J = diag(f'(W r* + bias)) @ W

    Returns
    -------
    J : np.ndarray, shape (N, N)
    d : np.ndarray, shape (N,) — diagonal gain vector f'(...)
    z : np.ndarray, shape (N,) — pre-activation W r* + bias
    """
    z = W @ r_star + bias
    d = fprime(z)
    return d[:, None] * W, d, z


def jacobians_for_all_fixed_points(W, biases, r_fps, dtype=np.float32):
    """
    Compute Jacobians at all K fixed points.

    Returns
    -------
    Js : np.ndarray, shape (N, N, K)
    ds : np.ndarray, shape (N, K)
    zs : np.ndarray, shape (N, K)
    """
    N, K = r_fps.shape
    Js = np.empty((N, N, K), dtype=dtype)
    ds = np.empty((N, K),    dtype=dtype)
    zs = np.empty((N, K),    dtype=dtype)
    for k in range(K):
        J, d, z = jacobian_at_rstar(W, biases[:, k], r_fps[:, k])
        Js[:, :, k] = J
        ds[:, k]    = d
        zs[:, k]    = z
    return Js, ds, zs


# =====================================================================
# Multi-orbit precomputation
# =====================================================================
def precompute_all_orbits(ic_orbits, W, tauE, tauI, T, dt, a, b, c,
                           r_stars, biases, no_freqs, wash):
    """
    Call precompute_orbit for each of the K bias conditions.

    Parameters
    ----------
    ic_orbits : np.ndarray, shape (N, K)
    r_stars   : np.ndarray, shape (N, K) — fixed points used as baselines (s)
    biases    : np.ndarray, shape (N, K)

    Returns
    -------
    orbit_infos : list of K orbit_info dicts (see utils.precompute_orbit)
    """
    from utils import precompute_orbit
    K = r_stars.shape[1]
    orbit_infos = [None] * K
    for k in range(K):
        orbit_infos[k] = precompute_orbit(
            ic_orbit=ic_orbits[:, k],
            W=W, tauE=tauE, tauI=tauI,
            T=T, dt=dt, a=a, b=b, c=c,
            s=r_stars[:, k],
            bias=biases[:, k],
            no_freqs=no_freqs,
            wash=wash,
        )
    return orbit_infos


# =====================================================================
# Controlled solver with bias switching, kicks, pulls, and OU noise
# =====================================================================
def _t_to_idx(t, dt):
    return int(np.round(float(t) / float(dt)))


def _clamp_idx(i, n):
    return max(0, min(int(i), n - 1))


def solve_rk4_bias_controlled(
    W, a, b, c,
    tauE, tauI,
    T, dt,
    biases,         # (N, K)
    r_stars,        # (N, K)
    kick_dirs,      # (N, K)
    orbit_infos,
    schedule,
    kick_gain=1, K_fp=10.0, resume_gain=10.0,
    k_norm=0.6, k_tan=0.15,
    sigma_ou=0.0, tau_c=0.01, t_start_noise=0.0, rng=None,
    k0=0, x0=None,
):
    """
    Simulate a long trajectory with scheduled bias switches, kicks, pulls,
    and phase-matched orbit resumes, plus optional OU noise.

    Schedule events (list of dicts):
      - {'kind': 'set_bias',      't':  float, 'k': int}
      - {'kind': 'kick_orbit',    't':  float, 'k': int, ['dur': float]}
      - {'kind': 'pull_fp',       't0': float, 't1': float, 'k': int}
      - {'kind': 'resume_orbit',  't0': float, 't1': float, 'k': int}

    Parameters
    ----------
    k0 : int — starting bias index
    x0 : np.ndarray or None — initial state (defaults to r_stars[:, k0])

    Returns
    -------
    tvec : np.ndarray, shape (Tn,)
    r    : np.ndarray, shape (Tn, N)
    """
    from utils import rk4_step, nearest_phase_idx

    if rng is None:
        rng = np.random.default_rng()

    N    = W.shape[0]
    tvec = np.arange(0.0, T + dt, dt)
    Tn   = len(tvec)

    x = np.zeros((N, Tn), dtype=float)
    x[:, 0] = r_stars[:, k0].copy() if x0 is None else np.array(x0, float).copy()

    quint = N // 5
    tau_vec = np.concatenate([
        float(tauE) * np.ones(4 * quint),
        float(tauI) * np.ones(N - 4 * quint)
    ])

    n = np.zeros(N, dtype=float)
    inv_tau_c = 1.0 / float(tau_c)
    if sigma_ou > 0:
        s_n = float(sigma_ou) * np.sqrt(float(tau_c) / 2.0)
        n   = rng.normal(0.0, s_n, size=N)

    # pre-index events
    events = []
    for ev in schedule:
        ev2 = dict(ev)
        if 't'  in ev2: ev2['idx'] = _clamp_idx(_t_to_idx(ev2['t'],  dt), Tn)
        if 't0' in ev2: ev2['i0']  = _clamp_idx(_t_to_idx(ev2['t0'], dt), Tn)
        if 't1' in ev2: ev2['i1']  = _clamp_idx(_t_to_idx(ev2['t1'], dt), Tn)
        events.append(ev2)

    k_current    = int(k0)
    bias_current = biases[:, k_current]

    for i in range(1, Tn):
        # bias switches
        for ev in events:
            if ev['kind'] == 'set_bias' and ev.get('idx') == (i - 1):
                k_current    = int(ev['k'])
                bias_current = biases[:, k_current]

        u = np.zeros(N, dtype=float)

        # kicks
        for ev in events:
            if ev['kind'] == 'kick_orbit':
                k = int(ev['k'])
                if 'dur' in ev:
                    i0 = ev['idx']
                    M  = max(1, int(np.round(ev['dur'] / dt)))
                    if i0 <= (i - 1) < i0 + M:
                        u += (kick_gain / (M * dt)) * kick_dirs[:, k]
                elif ev['idx'] == (i - 1):
                    u += (kick_gain / dt) * kick_dirs[:, k]

        # windows
        for ev in events:
            kind = ev['kind']
            if kind == 'pull_fp' and ev['i0'] <= (i - 1) < ev['i1']:
                u += K_fp * (r_stars[:, int(ev['k'])] - x[:, i - 1])

            elif kind == 'resume_orbit' and ev['i0'] <= (i - 1) < ev['i1']:
                k        = int(ev['k'])
                r_orbit  = orbit_infos[k]['r_orbit']
                t_hat    = orbit_infos[k]['t_hat']
                xi       = x[:, i - 1]
                j        = nearest_phase_idx(xi, r_orbit)
                r_star   = r_orbit[:, j]
                tauhat   = t_hat[:, j]
                e        = r_star - xi
                e_tan    = np.dot(e, tauhat) * tauhat
                e_norm   = e - e_tan
                u += resume_gain * (k_norm * e_norm + k_tan * tauhat)

        x_det = rk4_step(x[:, i - 1], W, u, tauE, tauI, dt, a, b, c, bias_current)

        tnow = tvec[i]
        if (tnow >= t_start_noise) and (sigma_ou > 0):
            n        = n + (-inv_tau_c * n) * dt + float(sigma_ou) * np.sqrt(dt) * rng.normal(size=N)
            x[:, i]  = x_det + dt * (n / tau_vec)
        else:
            x[:, i]  = x_det

    return tvec, x.T

# =====================================================================
# Stability analysis along paths between fixed points
#
# These functions support analysing how the spectral abscissa (max Re
# eigenvalue of the Jacobian) changes as the network is moved between
# two fixed points along a linear interpolation path.
#
# Two paths are compared:
#   - heterogeneous: r0 -> r1 (the actual biased fixed point)
#   - homogeneous:   r0 -> s*1 (a scalar-uniform control point at the
#                    same Euclidean distance as r1)
# =====================================================================

def f_inv_shifted_sigmoid(r):
    """
    Inverse of the shifted sigmoid f(x) = sigmoid(x) - 0.5:
        f^{-1}(r) = log((1 + 2r) / (1 - 2r))    for r in (-0.5, 0.5)
    """
    return np.log((1 + 2*r) / (1 - 2*r))


def bias_for_fixed_point(W, r):
    """
    Compute bias b such that r is an exact fixed point of -r + f(Wr + b) = 0:
        b(r) = f^{-1}(r) - W r
    """
    return f_inv_shifted_sigmoid(r) - W @ r


def D_from_r(r):
    """
    Diagonal gain at fixed point r for the shifted sigmoid:
        D_ii = f'(f^{-1}(r_i)) = 0.25 - r_i^2
    """
    return 0.25 - r**2


def jacobian_at_r(W, r):
    """
    Network Jacobian at state r (treated as a fixed point):
        J(r) = -I + diag(D(r)) @ W
    """
    return -np.eye(W.shape[0]) + D_from_r(r)[:, None] * W


def spectral_abscissa(J):
    """Maximum real part of the eigenvalues of J."""
    return float(np.max(np.linalg.eigvals(J).real))


def pick_s_matched_distance(r0, d_star):
    """
    Find scalar s such that ||s*1 - r0||_2 = d_star.
    Used to construct a homogeneous comparison point at the same distance
    as the heterogeneous target fixed point.

    Returns the feasible root in (-0.5, 0.5) farthest from mean(r0).
    """
    N   = r0.size
    a_c = N
    b_c = -2.0 * np.sum(r0)
    c_c = np.dot(r0, r0) - d_star**2
    disc = b_c**2 - 4*a_c*c_c
    if disc < 0:
        raise ValueError(f'No real s solves the distance match (disc={disc})')
    sqrt_disc = np.sqrt(disc)
    s1 = (-b_c + sqrt_disc) / (2*a_c)
    s2 = (-b_c - sqrt_disc) / (2*a_c)
    cand = [s for s in (s1, s2) if np.abs(s) < 0.5]
    if not cand:
        s = s1 if abs(s1) < abs(s2) else s2
        print(f'WARNING: matched-distance s={s:.4f} is outside (-0.5, 0.5)')
        return s
    cand.sort(key=lambda s: abs(s - np.mean(r0)), reverse=True)
    return cand[0]


def abscissa_curve_between(W, r0, rT, n=101):
    """
    Compute the spectral abscissa along the linear path from r0 to rT.

    At each interpolation step r(λ) = (1-λ)*r0 + λ*rT:
      - computes the bias that makes r(λ) an exact fixed point
      - computes the Jacobian J(r(λ)) and its spectral abscissa
      - records the distance from r0

    Points where r(λ) is outside (-0.5, 0.5) are marked NaN.

    Parameters
    ----------
    W      : np.ndarray, shape (N, N)
    r0, rT : np.ndarray, shape (N,) — start and end points
    n      : int — number of interpolation steps

    Returns
    -------
    dists  : np.ndarray, shape (n,)    — ||r(λ) - r0||
    alphas : np.ndarray, shape (n,)    — spectral abscissa (NaN if out of bounds)
    biases : np.ndarray, shape (N, n)  — bias at each point
    rs     : np.ndarray, shape (N, n)  — interpolated states
    Js     : np.ndarray, shape (N, N, n) — Jacobians
    """
    lam    = np.linspace(0, 1, n)
    alphas = np.full(n, np.nan)
    dists  = np.zeros(n)
    biases = np.full((W.shape[0], n), np.nan)
    rs     = np.full((W.shape[0], n), np.nan)
    Js     = np.full((W.shape[0], W.shape[1], n), np.nan)

    for i, l in enumerate(lam):
        r        = (1 - l) * r0 + l * rT
        rs[:, i] = r
        dists[i] = np.linalg.norm(r - r0)
        if np.any(np.abs(r) >= 0.5):
            continue                          # outside sigmoid's valid range
        biases[:, i] = bias_for_fixed_point(W, r)
        J            = jacobian_at_r(W, r)
        Js[:, :, i]  = J
        alphas[i]    = spectral_abscissa(J)

    return dists, alphas, biases, rs, Js
