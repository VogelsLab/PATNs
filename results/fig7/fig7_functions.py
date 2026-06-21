import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree


# =====================================================================
# Orbit distance metrics
# =====================================================================
def scale_only_normalize_orbit(X):
    """
    Scale-only normalization: divide by orbit size S (no mean subtraction).

    Returns
    -------
    X_scaled : np.ndarray
    S        : float
    """
    X = np.asarray(X, float)
    T, d = X.shape
    center = X.mean(axis=0)
    Xc = X - center
    cov = np.cov(Xc, rowvar=False)
    eigvals, Q = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, Q = eigvals[order], Q[:, order]
    r = np.sqrt(np.maximum(eigvals, 0.0))
    Z = Xc @ Q
    max_extent = np.max(np.abs(Z), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        scale_per_axis = np.where(r > 1e-12, max_extent / r, 1.0)
    scale_base = max(float(np.max(scale_per_axis)), 1.0)
    a_axes = r * scale_base
    S = float(np.sqrt(np.mean(a_axes**2)))
    S = max(S, 1e-12)
    return X / S, S

def directed_p95_kdtree(X_ref, X_test, q=95):
    """
    Directed pq NN distance from X_test -> X_ref.

    Returns
    -------
    pq    : float
    dists : np.ndarray, shape (len(X_test),)
    """
    tree = cKDTree(X_ref)
    dists, _ = tree.query(X_test, k=1)
    return float(np.percentile(dists, q)), dists


def pairwise_orbit_distances_scale_only(orbits, q=95):
    """
    Compute pairwise orbit distances after scale-only normalization.

    Each orbit Xi is divided by its own size Si (no mean subtraction).
    Then directed and symmetric p_q NN distances are computed.

    Parameters
    ----------
    orbits : list of np.ndarray, each shape (T_k, N)
    q      : int — percentile (default 95)

    Returns
    -------
    dict with:
        'S'              : np.ndarray, shape (K,) — orbit sizes
        'p_q_dir_scaled' : np.ndarray, shape (K, K) — directed distances
        'sym_scaled'     : np.ndarray, shape (K, K) — symmetric (worst-case)
    """
    K = len(orbits)
    orbits_scaled = []
    S = np.zeros(K, float)
    for i, X in enumerate(orbits):
        Xs, Si = scale_only_normalize_orbit(X)
        orbits_scaled.append(Xs)
        S[i] = Si

    p_q = np.zeros((K, K), float)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            pq, _ = directed_p95_kdtree(orbits_scaled[i], orbits_scaled[j], q=q)
            p_q[i, j] = pq  # j -> i

    return {'S': S, 'p_q_dir_scaled': p_q, 'sym_scaled': np.maximum(p_q, p_q.T)}


# =====================================================================
# Color helpers
# =====================================================================
def cycle_color(k, cmap=None):
    """Consistent color per orbit index k."""
    if cmap is None:
        cmap = plt.get_cmap('tab20b')
    return 'k' if k == 0 else cmap(k + 11)


def cycle_color_perm(k, perm=None, cmap=None):
    """Color after applying index permutation."""
    if perm is None:
        perm = [0, 1, 4, 3, 2]
    return cycle_color(perm[k], cmap=cmap)


# =====================================================================
# 2D readout trajectory plotting
# =====================================================================
def ensure_TN(X, N=100):
    """Accept (T,N) or (N,T) and return (T,N)."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError('Expected 2D array')
    if X.shape[1] == N:
        return X
    if X.shape[0] == N:
        return X.T
    raise ValueError(f'Cannot infer orientation for shape {X.shape} with N={N}')


def plot_line_gradient(signal, ax=None, cmap='viridis', lw=2, alpha=1.0):
    """Plot a 2D polyline with a time-varying color gradient."""
    signal = np.asarray(signal, float)
    if signal.ndim != 2 or signal.shape[1] != 2:
        raise ValueError('Expected shape (T, 2)')
    if ax is None:
        ax = plt.gca()
    pts  = signal.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    if len(segs) < 1:
        return None
    tvals = np.linspace(0, 1, len(segs))
    rgba  = cm.get_cmap(cmap)(tvals)
    rgba[:, 3] = alpha
    lc = LineCollection(segs, colors=rgba, linewidth=lw)
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    return lc


def plot_line_constant(signal, ax=None, color='0.6', lw=2, alpha=0.35):
    """Plot a 2D polyline with a constant color."""
    signal = np.asarray(signal, float)
    if signal.ndim != 2 or signal.shape[1] != 2:
        raise ValueError('Expected shape (T, 2)')
    if ax is None:
        ax = plt.gca()
    ax.plot(signal[:, 0], signal[:, 1], color=color, lw=lw, alpha=alpha)
    ax.set_aspect('equal', adjustable='box')


# =====================================================================
# Segment construction from transition times
# =====================================================================
def segments_from_transition_times(t_list, dt, T_total):
    """
    Build (i0, i1) index pairs from a list of transition times.

    Parameters
    ----------
    t_list  : list of float — transition times in seconds
    T_total : int or float — total time in samples (int) or seconds (float)

    Returns
    -------
    segs : list of (int, int)
    """
    T_total = int(T_total / dt) if isinstance(T_total, float) else int(T_total)
    idx  = [max(0, min(T_total, int(np.round(t / dt)))) for t in t_list]
    idx  = sorted(set(idx))
    cuts = [0] + idx + [T_total]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]


# =====================================================================
# Cycle segmentation helpers
# =====================================================================
def estimate_cycle_length_2d_by_lag(Y, Lmin, Lmax):
    """
    Estimate cycle length L in [Lmin, Lmax] by minimizing ||Y[t] - Y[t+L]||².
    """
    Y    = np.asarray(Y, float)
    T    = Y.shape[0]
    Lmin = int(max(2, Lmin))
    Lmax = int(min(T - 2, Lmax))
    if T < Lmax + 5 or Lmax <= Lmin:
        return max(2, min(int(Lmin), T // 2))
    Yc = Y - Y.mean(axis=0, keepdims=True)
    bestL, bestScore = Lmin, np.inf
    for L in range(Lmin, Lmax + 1):
        D     = Yc[:-L] - Yc[L:]
        score = np.mean(np.sum(D * D, axis=1))
        if score < bestScore:
            bestScore, bestL = score, L
    return int(bestL)


def refine_cut_near_return(Y, k, L, window=25, min_sep=5):
    """
    From index k, search near k+L for the point closest to Y[k].
    Returns refined cut index or None.
    """
    Y     = np.asarray(Y, float)
    T     = Y.shape[0]
    k_nom = k + int(L)
    if k_nom >= T:
        return None
    a = max(k + min_sep, k_nom - int(window))
    b = min(T - 1, k_nom + int(window))
    if b <= a:
        return k_nom
    d2 = np.sum((Y[a:b] - Y[k])**2, axis=1)
    return int(a + np.argmin(d2))


def cuts_from_cycle_length(Y, L, refine_window=25, min_cycle_len=20):
    """
    Repeatedly jump ~L ahead (with refinement) to find cycle boundaries.

    Returns
    -------
    cuts : list of int — monotonically increasing, including 0 and T-1
    """
    Y = np.asarray(Y, float)
    T = Y.shape[0]
    if T < 2:
        return [0, T - 1]
    cuts = [0]; k = 0
    while True:
        k_next = refine_cut_near_return(Y, k, L, window=refine_window)
        if k_next is None or k_next <= k + min_cycle_len or k + L >= T - 2:
            break
        cuts.append(k_next); k = k_next
    if cuts[-1] != T - 1:
        cuts.append(T - 1)
    cuts = sorted(set(int(c) for c in cuts if 0 <= c <= T - 1))
    return cuts if len(cuts) >= 2 else [0, T - 1]


# =====================================================================
# Main readout plot: segments split into individual cycle figures
# =====================================================================
def plot_segments_cycles_separately(
        Y_long, segs, segment_labels, dt,
        orbit_transient_len=None, fp_transient_len=None,
        cycle_period_samples=(105, 125), cycle_refine_window=25,
        min_cycle_len=20, cmap='viridis',
        lw=2, lw_transient=1.5, alpha_steady=1.0,
        transient_color='0.6', alpha_transient=0.35,
        xlim=(-3, 10), ylim=(-2, 2),
        figsize=(4, 1), dpi=500,
        show_transient_panel_for_orbits=True):
    """
    Plot each segment of the readout trajectory separately:
    - Fixed-point segments: gradient (with optional gray transient)
    - Orbit segments: each cycle repetition in its own figure

    Parameters
    ----------
    Y_long          : np.ndarray, shape (T, 2)
    segs            : list of (i0, i1) from segments_from_transition_times
    segment_labels  : list of str, e.g. ['FP0','O0','O1','FP1',...]
    orbit_transient_len : dict label -> int samples to treat as transient
    fp_transient_len    : dict label -> int samples to treat as transient
    """
    if orbit_transient_len is None:
        orbit_transient_len = {}
    if fp_transient_len    is None:
        fp_transient_len    = {}

    def _get_L(label, dct, prefix):
        L = dct.get(label)
        if L is None:
            try:
                L = dct.get(int(label[len(prefix):]), 0)
            except Exception:
                L = 0
        return int(max(0, L))

    def _style(ax):
        for side in ('top', 'right', 'left', 'bottom'):
            ax.spines[side].set_visible(False)
        for s in ax.spines.values():
            s.set_linewidth(0.45)
        ax.tick_params(width=0.45, pad=1, length=2, labelsize=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    def _new_fig():
        return plt.subplots(figsize=figsize, dpi=dpi)

    for (i0, i1), label in zip(segs, segment_labels):
        Yseg = np.asarray(Y_long[i0:i1], float)
        T    = Yseg.shape[0]
        if T < 2:
            continue

        if label.startswith('FP'):
            Lfp = min(_get_L(label, fp_transient_len, 'FP'), T)
            fig, ax = _new_fig()
            if Lfp >= 2:
                plot_line_constant(0.7 * Yseg[:Lfp], ax=ax, color=transient_color,
                                   lw=lw_transient, alpha=alpha_transient)
            Yst = Yseg[Lfp:] if T - Lfp >= 2 else Yseg
            plot_line_gradient(Yst, ax=ax, cmap=cmap, lw=lw, alpha=alpha_steady)
            _style(ax); plt.show()

        elif label.startswith('O'):
            Ltr = min(_get_L(label, orbit_transient_len, 'O'), T)
            Ytr = 0.5 * Yseg[:Ltr] if Ltr >= 2 else None
            Yst = Yseg[Ltr:]        if T - Ltr >= 2 else None

            if show_transient_panel_for_orbits and Ytr is not None:
                fig, ax = _new_fig()
                plot_line_constant(Ytr, ax=ax, color=transient_color,
                                   lw=lw_transient, alpha=alpha_transient)
                _style(ax); plt.show()

            if Yst is not None and len(Yst) >= 4:
                Lmin_s, Lmax_s = cycle_period_samples
                Lcyc  = estimate_cycle_length_2d_by_lag(Yst, Lmin=Lmin_s, Lmax=Lmax_s)
                cuts  = cuts_from_cycle_length(Yst, L=Lcyc,
                                               refine_window=cycle_refine_window,
                                               min_cycle_len=min_cycle_len)
                for a, b in zip(cuts[:-1], cuts[1:]):
                    Ycycle = Yst[a:b + 1]
                    if len(Ycycle) < 2:
                        continue
                    fig, ax = _new_fig()
                    plot_line_gradient(Ycycle, ax=ax, cmap=cmap, lw=lw, alpha=alpha_steady)
                    _style(ax); plt.show()

        else:
            fig, ax = _new_fig()
            plot_line_gradient(Yseg, ax=ax, cmap=cmap, lw=lw, alpha=alpha_steady)
            _style(ax); plt.show()


# =====================================================================
# Bias-over-time plot
# =====================================================================
def plot_bias_over_time(biases, transition_times_ordered, dt,
                        T_total=None, segment_bias_ids=None,
                        ax=None, lw=0.6, alpha=0.6, show_vlines=False):
    """
    Plot per-neuron bias as lines that jump at transition times.

    Parameters
    ----------
    biases                    : np.ndarray, shape (N, 5)
    transition_times_ordered  : list of 8 floats (seconds) — 9 segments
    segment_bias_ids          : list of 9 ints in {0..4}
    T_total                   : int (samples) or None

    Returns
    -------
    ax, B, bounds, idx_trans
    """
    biases = np.asarray(biases)
    assert biases.ndim == 2 and biases.shape[1] == 5, 'biases must be (N, 5)'
    N = biases.shape[0]

    if segment_bias_ids is None:
        segment_bias_ids = [0, 0, 1, 1, 2, 4, 3, 4, 0]
    assert len(segment_bias_ids) == 9
    assert all(0 <= k <= 4 for k in segment_bias_ids)

    t_trans   = np.asarray(transition_times_ordered, float)
    assert t_trans.size == 8
    idx_trans = np.rint(t_trans / dt).astype(int)

    if np.any(np.diff(idx_trans) < 0):
        raise ValueError('transition_times_ordered must be nondecreasing')
    if T_total is None:
        T_total = int(idx_trans[-1]) + 1

    bounds = np.concatenate(([0], idx_trans, [T_total]))
    B = np.empty((T_total, N), dtype=float)
    for s in range(9):
        i0, i1 = bounds[s], bounds[s + 1]
        k = segment_bias_ids[s]
        if i1 > i0:
            B[i0:i1, :] = biases[:, k][None, :]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 0.3), dpi=500)

    t = np.arange(T_total) * dt
    ax.plot(t, B[:, 23], 'gray', linewidth=0.2, alpha=0.4)
    ax.plot(t, B[:, 21], 'gray', linewidth=0.2, alpha=0.4)
    ax.plot(t, B[:, 57], 'gray', linewidth=0.2, alpha=0.4)
    ax.plot(t, B[:, 66], 'gray', linewidth=0.2, alpha=0.4)
    ax.plot(t, B[:,  9], color='gray', linewidth=0.6, alpha=0.8)
    ax.plot(t, B[:, 74], color='gray', linewidth=0.6, alpha=0.7)

    if show_vlines:
        for tt in t_trans:
            ax.axvline(tt, lw=0.8, alpha=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 15)

    return ax, B, bounds, idx_trans
