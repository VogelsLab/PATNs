import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from skimage import io, color, filters, measure, morphology


# =====================================================================
# Noisy simulation
# =====================================================================
def simulate_rk4_with_ou_noise(W, a, b, c, tauE, tauI, T, dt, bias, x0,
                                sigma_ou=0.0, tau_c=0.01, t_start_noise=0.0,
                                rng=None, ou_init='steady', burn_in=0.0):
    """
    RK4 dynamics with OU noise injection:
        x_det  = rk4_step(x, W, 0, tauE, tauI, dt, a, b, c, bias)
        n     += (-n/tau_c)*dt + sigma_ou*sqrt(dt)*N(0,I)
        x_next = x_det + dt*(n/tau_vec)

    Parameters
    ----------
    burn_in : float — seconds to simulate before t=0 (not returned)
    ou_init : str   — 'steady' (initialize OU at stationary std) or 'zero'

    Returns
    -------
    tvec : np.ndarray, shape (Tn,)
    X    : np.ndarray, shape (Tn, N)
    n    : np.ndarray, shape (N,) — final OU state
    """
    from utils import rk4_step

    if rng is None:
        rng = np.random.default_rng()

    bias = np.asarray(bias, float)
    x0   = np.asarray(x0,   float)
    N    = W.shape[0]

    quint   = N // 5
    tau_vec = np.concatenate([
        float(tauE) * np.ones(4 * quint),
        float(tauI) * np.ones(N - 4 * quint)
    ])
    inv_tau_c = 1.0 / float(tau_c)
    u = np.zeros(N, dtype=float)

    n = np.zeros(N, dtype=float)
    if sigma_ou > 0 and ou_init == 'steady':
        s_n = float(sigma_ou) * np.sqrt(float(tau_c) / 2.0)
        n   = rng.normal(0.0, s_n, size=N)

    def step_one(x, tnow, n):
        x_det = rk4_step(x, W, u, tauE, tauI, dt, a, b, c, bias)
        if (tnow >= t_start_noise) and (sigma_ou > 0):
            n      = n + (-inv_tau_c * n) * dt + float(sigma_ou) * np.sqrt(dt) * rng.normal(size=N)
            x_next = x_det + dt * (n / tau_vec)
        else:
            x_next = x_det
        return x_next, n

    x    = x0.copy()
    tnow = 0.0
    if burn_in and burn_in > 0:
        for _ in range(int(np.round(burn_in / dt))):
            x, n = step_one(x, tnow, n)
            tnow += dt

    Tn   = int(np.round(T / dt)) + 1
    tvec = dt * np.arange(Tn)
    X    = np.zeros((Tn, N), dtype=float)
    X[0] = x
    for i in range(1, Tn):
        X[i], n = step_one(X[i - 1], tvec[i], n)

    return tvec, X, n


# =====================================================================
# Shape utilities
# =====================================================================
def center_and_scale(Y, scale=1.0):
    """Center Y and normalize to unit RMS amplitude."""
    Y   = np.asarray(Y, float)
    Y   = Y - Y.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(Y**2, axis=1)))
    return Y if rms < 1e-12 else (scale / rms) * Y


def rotate(Y, angle_rad):
    """Rotate a (T,2) curve by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return Y @ np.array([[c, -s], [s, c]]).T


def resample_closed_curve(Y, n):
    """Resample a CLOSED curve Y (T,2) to n equally-spaced arc-length points."""
    Y  = np.asarray(Y, float)
    Yc = np.vstack([Y, Y[0]])
    d  = np.sqrt(np.sum(np.diff(Yc, axis=0)**2, axis=1))
    s  = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-12:
        return np.repeat(Y[:1], n, axis=0)
    s_new = np.linspace(0, s[-1], n + 1)[:-1]
    out   = np.zeros((n, 2))
    for dim in range(2):
        out[:, dim] = np.interp(s_new, s, Yc[:, dim])
    return out


def chaikin_closed(pts, n_iter=3):
    """Chaikin corner-cutting for a CLOSED polyline."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return P
    if np.linalg.norm(P[0] - P[-1]) > 1e-12:
        P = np.vstack([P, P[0]])
    for _ in range(n_iter):
        Q = []
        for i in range(len(P) - 1):
            p0, p1 = P[i], P[i + 1]
            Q.append(0.75 * p0 + 0.25 * p1)
            Q.append(0.25 * p0 + 0.75 * p1)
        P = np.vstack(Q + [Q[0]])
    return P[:-1]


def curve_from_icon_png(path, n=400, scale=1.0, angle=0.0,
                        invert=False, blur_sigma=1.0, threshold=None,
                        min_size=200, smooth_iter=0):
    """
    Convert a silhouette image (dark on light background) to a closed 2D curve.

    Parameters
    ----------
    path        : str — path to image file
    n           : int — number of output points
    scale       : float — output scale
    angle       : float — rotation angle in radians
    invert      : bool — invert mask (for light on dark)
    blur_sigma  : float — Gaussian blur before thresholding
    threshold   : float or None — manual threshold (None = Otsu)
    min_size    : int — minimum connected component size
    smooth_iter : int — Chaikin smoothing iterations

    Returns
    -------
    Y : np.ndarray, shape (n, 2)
    """
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(float)
    if blur_sigma and blur_sigma > 0:
        img = filters.gaussian(img, sigma=blur_sigma)
    if threshold is None:
        threshold = filters.threshold_otsu(img)
    mask = img < threshold
    if invert:
        mask = ~mask
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.binary_closing(mask, morphology.disk(2))
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        raise ValueError(f'No contour found in {path}')
    c = max(contours, key=lambda a: a.shape[0])
    Y = np.column_stack([c[:, 1], -c[:, 0]])
    Y = resample_closed_curve(Y, n)
    if smooth_iter > 0:
        Y = chaikin_closed(Y, n_iter=smooth_iter)
        Y = resample_closed_curve(Y, n)
    Y = center_and_scale(Y, scale=scale)
    return rotate(Y, angle)


def generate_shape_for_T(shape_name, T, shape_cache, shape_specs):
    """
    Generate a target shape with exactly T points. Results are cached.

    Parameters
    ----------
    shape_cache : dict — mutable cache dict (shared across calls)
    shape_specs : dict — SHAPE_SPECS dict mapping name -> curve_from_icon_png kwargs
    """
    key = (shape_name, int(T))
    if key in shape_cache:
        return shape_cache[key]
    spec = shape_specs[shape_name]
    Y    = curve_from_icon_png(
        path=spec['path'], n=int(T),
        scale=spec.get('scale', 1), angle=spec.get('angle', 0.0),
        invert=spec.get('invert', False), blur_sigma=spec.get('blur_sigma', 1.0),
        threshold=spec.get('threshold', None), min_size=spec.get('min_size', 200),
        smooth_iter=spec.get('smooth_iter', 0))
    shape_cache[key] = Y
    return Y


# =====================================================================
# Scoring and plotting
# =====================================================================
def r2_score(Y, Yhat, eps=1e-12):
    """
    Per-dimension and mean R² score.

    Returns
    -------
    r2_dim  : np.ndarray, shape (2,)
    r2_mean : float
    """
    Y, Yhat = np.asarray(Y), np.asarray(Yhat)
    ss_res  = np.sum((Y - Yhat)**2, axis=0)
    ss_tot  = np.sum((Y - Y.mean(axis=0, keepdims=True))**2, axis=0) + eps
    r2_dim  = 1.0 - ss_res / ss_tot
    return r2_dim, float(np.mean(r2_dim))


def plot_shape_fit(Y, Yhat, title='', show_target=True):
    """Plot target vs decoded 2D shape."""
    plt.figure()
    if show_target:
        plt.plot(Y[:, 0], Y[:, 1], 'k.', alpha=0.25, label='target')
    plt.plot(Yhat[:, 0], Yhat[:, 1], 'r-', linewidth=2, label='decoded')
    plt.axis('equal')
    plt.legend()
    plt.title(title)
    plt.show()


# =====================================================================
# Assignment fitting
# =====================================================================
def fit_assignment(assignment, orbit_infos, orbit_ks,
                   shape_cache, shape_specs,
                   show_target=True, plot=False):
    """
    Fit a Ridge readout for one assignment of shapes to orbits.

    Parameters
    ----------
    assignment  : dict {k: shape_name}
    orbit_ks    : list of int — orbit indices to fit
    shape_cache : dict — mutable shape cache
    shape_specs : dict — SHAPE_SPECS

    Returns
    -------
    res : dict with 'assignment', 'model', 'r2_mean_all', 'per_orbit', etc.
    """
    Xs, Ys, meta = [], [], []
    start = 0
    for k in orbit_ks:
        r_orbit    = np.asarray(orbit_infos[k]['r_orbit'], float).T  # (T_k, N)
        T          = r_orbit.shape[0]
        Y          = generate_shape_for_T(assignment[k], T, shape_cache, shape_specs)
        Xs.append(r_orbit); Ys.append(Y)
        meta.append((k, assignment[k], start, start + T, T))
        start += T

    X_all  = np.vstack(Xs)
    Y_all  = np.vstack(Ys)
    model  = Ridge(alpha=0, fit_intercept=True)
    model.fit(X_all, Y_all)
    Yhat   = model.predict(X_all)
    r2_dim_all, r2_mean_all = r2_score(Y_all, Yhat)

    per_orbit = {}
    for (k, shape_name, a, b, T) in meta:
        r2_dim, r2_mean = r2_score(Y_all[a:b], Yhat[a:b])
        per_orbit[k] = dict(shape=shape_name, r2_dim=r2_dim, r2_mean=r2_mean)

    res = dict(assignment=assignment, model=model,
               X_all=X_all, Y_all=Y_all, Yhat_all=Yhat, meta=meta,
               r2_dim_all=r2_dim_all, r2_mean_all=r2_mean_all, per_orbit=per_orbit)

    if plot:
        for (k, shape_name, a, b, T) in meta:
            plot_shape_fit(Y_all[a:b], Yhat[a:b],
                           title=f'k={k}  shape={shape_name}  R2={per_orbit[k]["r2_mean"]:.4f}',
                           show_target=show_target)
    return res


def run_all_permutations(orbit_infos, orbit_ks, shape_names,
                         shape_cache, shape_specs):
    """
    Try all len(shape_names)! permutations of shapes onto orbits.
    Returns list of compact result dicts sorted by R² descending.
    """
    results = []
    for perm in itertools.permutations(shape_names, len(orbit_ks)):
        assignment = {k: perm[i] for i, k in enumerate(orbit_ks)}
        res = fit_assignment(assignment, orbit_infos, orbit_ks,
                             shape_cache, shape_specs, plot=False)
        results.append(dict(assignment=assignment,
                            r2_mean_all=res['r2_mean_all'],
                            r2_dim_all=res['r2_dim_all'],
                            per_orbit=res['per_orbit']))
    results.sort(key=lambda d: d['r2_mean_all'], reverse=True)
    return results


def assignment_str(assignment, orbit_ks):
    return ' | '.join([f'k{k}:{assignment[k]}' for k in orbit_ks])


def print_top(results, orbit_ks, top=10):
    """Print the top-ranked permutation assignments."""
    for i in range(min(top, len(results))):
        r = results[i]
        print(f'{i+1:3d}) R2={r["r2_mean_all"]:.6f}   {assignment_str(r["assignment"], orbit_ks)}')


def plot_rank(results, rank, orbit_infos, orbit_ks, shape_cache, shape_specs,
              show_target=True):
    """Re-fit and plot the rank-th best assignment (1-based)."""
    r = results[rank - 1]
    print(f'Rank {rank}: R2={r["r2_mean_all"]:.6f}')
    print(assignment_str(r['assignment'], orbit_ks))
    fit_assignment(r['assignment'], orbit_infos, orbit_ks,
                   shape_cache, shape_specs, show_target=show_target, plot=True)


# =====================================================================
# Training set construction
# =====================================================================
def build_weighted_training_set_noisy(
        orbit_infos, best_assignment, W, a, b, c, tauE, tauI, dt, biases,
        shape_cache, shape_specs,
        orbit_ks=(0, 1, 2, 3, 4),
        n_noisy_per_orbit=10, seed0=0, burn_in=0,
        sigma_ou=0.0, tau_c=0.01, t_start_noise=0.0, ou_init='steady',
        w_det=1.0, w_noisy=0.3):
    """
    Build training data: one deterministic example + n_noisy_per_orbit noisy
    simulations per orbit.

    Returns
    -------
    X_all, Y_all, w_all : np.ndarray
    meta                : list of dicts
    """
    X_chunks, Y_chunks, w_chunks, meta = [], [], [], []
    for k in orbit_ks:
        r_orbit = np.asarray(orbit_infos[k]['r_orbit'], float)  # (N, T_k)
        N, T_k  = r_orbit.shape
        Yk      = generate_shape_for_T(best_assignment[k], T_k, shape_cache, shape_specs)

        X_chunks.append(r_orbit.T)
        Y_chunks.append(Yk)
        w_chunks.append(np.full(T_k, w_det, dtype=float))
        meta.append(dict(kind='det', k=k, shape=best_assignment[k], T=T_k))

        x0    = r_orbit[:, 0].copy()
        T_sim = (T_k - 1) * dt
        for j in range(n_noisy_per_orbit):
            rng = np.random.default_rng(seed0 + 100 * k + j)
            tvec, X_noisy, _ = simulate_rk4_with_ou_noise(
                W=W, a=a, b=b, c=c, tauE=tauE, tauI=tauI,
                T=T_sim, dt=dt, bias=biases[:, k], x0=x0,
                sigma_ou=sigma_ou, tau_c=tau_c,
                t_start_noise=t_start_noise, rng=rng,
                ou_init=ou_init, burn_in=burn_in)
            X_chunks.append(X_noisy)
            Y_chunks.append(Yk)
            w_chunks.append(np.full(T_k, w_noisy, dtype=float))
            meta.append(dict(kind='noisy', k=k, shape=best_assignment[k],
                             T=T_k, seed=int(seed0 + 1000 * k + j)))

    return np.vstack(X_chunks), np.vstack(Y_chunks), np.concatenate(w_chunks), meta


def augment_training_with_noisy_fixed_points(
        X_all, Y_all, w_all, r_stars, biases,
        ks_fp=(0, 1, 4), fp_targets=None,
        W=None, a=None, b=None, c=None, tauE=None, tauI=None,
        dt=0.005, T_fp=0.5, burn_in=0, sigma_ou=0.003, tau_c=0.01,
        ou_init='steady', t_start_noise=0.0, n_rollouts=10, seed0=0,
        w_fp=0.05, stride=5):
    """
    Augment training data with noisy fixed-point rollouts.

    Returns
    -------
    X_aug, Y_aug, w_aug : np.ndarray
    meta                : list of dicts (FP chunks only)
    """
    if fp_targets is None:
        fp_targets = {k: np.array([0.0, 0.0]) for k in ks_fp}

    X_chunks = [X_all]; Y_chunks = [Y_all]; w_chunks = [w_all]
    meta = []

    for k in ks_fp:
        x0 = np.asarray(r_stars[:, k], float).copy()
        yk = np.asarray(fp_targets[k], float).reshape(1, 2)
        for j in range(n_rollouts):
            rng = np.random.default_rng(seed0 + 1000 * k + j)
            tvec, X_fp, _ = simulate_rk4_with_ou_noise(
                W=W, a=a, b=b, c=c, tauE=tauE, tauI=tauI,
                T=T_fp, dt=dt, bias=biases[:, k], x0=x0,
                sigma_ou=sigma_ou, tau_c=tau_c,
                t_start_noise=t_start_noise, rng=rng,
                ou_init=ou_init, burn_in=burn_in)
            X_fp = X_fp[::stride]
            Y_fp = np.repeat(yk, X_fp.shape[0], axis=0)
            X_chunks.append(X_fp); Y_chunks.append(Y_fp)
            w_chunks.append(np.full(X_fp.shape[0], w_fp, float))
            meta.append(dict(kind='fp_noisy', k=int(k), rollout=int(j), n=X_fp.shape[0]))

    return np.vstack(X_chunks), np.vstack(Y_chunks), np.concatenate(w_chunks), meta
