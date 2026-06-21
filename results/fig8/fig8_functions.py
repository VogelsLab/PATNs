"""
fig8_functions.py
=================
Functions for Figure 8: subspace dimensionality, cross-projection, and tangling.

Cross-projection method: Elsayed et al. 2016
    "Reorganization of the motor cortex for control of novel movements"
    Each block's top-k PCs are computed; the cross-projection score from
    block j onto block i is the fraction of block j's variance explained
    by block i's k-dimensional subspace.

Tangling method: Russo & Churchland 2018
    "Motor cortex embeds muscle-like commands in an untangled space"
    Q(t) = max_{t': condition[t'] != condition[t]}
               ||x_dot(t) - x_dot(t')||^2 / (||x(t) - x(t')||^2 + epsilon)
    where epsilon = alpha * sum_k Var(X_k).
"""

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


# =====================================================================
# Trajectory utilities
# =====================================================================
def linear_resample_trajectory(r, n_samples):
    """
    Resample trajectory r of shape (T, N) to n_samples points via
    linear interpolation along normalized time.
    """
    T, N = r.shape
    if T == n_samples:
        return r.copy()
    if T < 2:
        return np.repeat(r, n_samples, axis=0)
    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, n_samples)
    r_new = np.empty((n_samples, N))
    for j in range(N):
        r_new[:, j] = np.interp(t_new, t_old, r[:, j])
    return r_new


def first_entry_and_stay_time(r, target_fp, threshold):
    """
    Return the first index i such that ||r[k] - target_fp|| < threshold
    for all k >= i (i.e. the trajectory stably enters the FP neighbourhood).

    Returns None if the trajectory never stably enters.
    """
    dists  = np.linalg.norm(r - target_fp[None, :], axis=1)
    inside = dists < threshold

    suffix_all_true       = np.zeros_like(inside, dtype=bool)
    suffix_all_true[-1]   = inside[-1]
    for i in range(len(inside) - 2, -1, -1):
        suffix_all_true[i] = inside[i] and suffix_all_true[i + 1]

    idx = np.where(suffix_all_true)[0]
    return None if len(idx) == 0 else idx[0]


def normalize_trajectory_block(block, mode='rms', eps=1e-12):
    """
    Mean-center and scale one trajectory block of shape (T, N).

    Parameters
    ----------
    mode : str — 'rms' | 'fro' | 'maxnorm'

    Returns
    -------
    block_norm : np.ndarray, shape (T, N)
    scale      : float
    """
    block_centered = block - block.mean(axis=0, keepdims=True)
    if mode == 'rms':
        scale = np.sqrt(np.mean(block_centered**2))
    elif mode == 'fro':
        scale = np.linalg.norm(block_centered)
    elif mode == 'maxnorm':
        scale = np.max(np.linalg.norm(block_centered, axis=1))
    else:
        raise ValueError("mode must be 'rms', 'fro', or 'maxnorm'")
    if scale < eps:
        scale = 1.0
    return block_centered / scale, scale


def build_dataset_from_blocks(blocks, labels_per_block=None):
    """
    Stack a list of trajectory blocks (T_i, N) into one dataset.

    Returns
    -------
    X      : np.ndarray, shape (sum T_i, N)
    labels : np.ndarray or None
    """
    X = np.vstack(blocks)
    labels = None
    if labels_per_block is not None:
        labels = np.array([lab for block, lab in zip(blocks, labels_per_block)
                           for _ in range(len(block))])
    return X, labels


def run_pca_trajectory_normalized(blocks, labels_per_block=None,
                                  norm_mode='rms', n_components=None):
    """
    Per-block RMS normalization followed by pooled PCA.

    Each block is mean-centered and scaled independently before pooling,
    so blocks with different durations contribute comparably.

    Returns
    -------
    dict with keys:
        'X_processed'            : (sum T_i, N) pooled normalized array
        'scores'                 : PCA scores
        'components'             : PCA components
        'explained_variance_ratio': np.ndarray
        'cum_explained_variance' : np.ndarray
        'block_scales'           : np.ndarray, shape (n_blocks,)
        'normalized_blocks'      : list of normalized arrays
        'labels'                 : np.ndarray or None
        'pca'                    : fitted PCA object
    """
    norm_blocks, scales = [], []
    for block in blocks:
        b_norm, scale = normalize_trajectory_block(block, mode=norm_mode)
        norm_blocks.append(b_norm)
        scales.append(scale)

    Xn, labels = build_dataset_from_blocks(norm_blocks, labels_per_block)
    Xc  = Xn - Xn.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xc)

    return {
        'X_processed':             Xc,
        'scores':                  scores,
        'components':              pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cum_explained_variance':  np.cumsum(pca.explained_variance_ratio_),
        'block_scales':            np.array(scales),
        'normalized_blocks':       norm_blocks,
        'labels':                  labels,
        'pca':                     pca,
    }


def individual_trajectory_dimensionalities(blocks, thresh=0.90):
    """
    Compute the number of PCs needed to explain `thresh` variance
    for each block individually.

    Returns
    -------
    dims : np.ndarray, shape (n_blocks,) — n90 per block
    """
    dims = []
    for block in blocks:
        bc  = block - block.mean(axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(bc, full_matrices=False)
        ve  = np.cumsum(S**2) / np.sum(S**2)
        dims.append(int(np.searchsorted(ve, thresh) + 1))
    return np.array(dims)


def pooled_n90(blocks):
    """
    Run normalized PCA on a list of blocks and return (cev, n90, pca_result).
    """
    labels = [f'b{i}' for i in range(len(blocks))]
    pca    = run_pca_trajectory_normalized(blocks, labels_per_block=labels, norm_mode='rms')
    cev    = pca['cum_explained_variance']
    n90    = int(np.searchsorted(cev, 0.90) + 1)
    return cev, n90, pca


def all_block_cev(blocks, max_pcs=60):
    """
    Per-block cumulative explained variance curves, padded to max_pcs.

    Returns
    -------
    cev_matrix : np.ndarray, shape (n_blocks, max_pcs)
    """
    out = []
    for b in blocks:
        bc = b - b.mean(axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(bc, full_matrices=False)
        ve = np.cumsum(S**2) / np.sum(S**2)
        if len(ve) < max_pcs:
            ve = np.concatenate([ve, np.ones(max_pcs - len(ve))])
        out.append(ve[:max_pcs])
    return np.array(out)


# =====================================================================
# Transient simulation and processing
# =====================================================================
def simulate_and_process_transient(ic, target_fp, label, idx_ic,
                                   W, tauE, tauI, T, dt, a, b, c, bias,
                                   threshold, min_raw_length=50,
                                   max_raw_length=500):
    """
    Simulate a transient trajectory and truncate at fixed-point arrival.

    The trajectory is truncated at the first time step from which it stays
    permanently within `threshold` of `target_fp`. If the trajectory never
    reaches the FP, the full trajectory is kept. Short transients
    (< min_raw_length) are discarded; long ones are capped at max_raw_length.

    Returns
    -------
    r_keep : np.ndarray, shape (T_kept, N) or None if too short
    meta   : dict or None
    """
    from utils import solve_rk4_springBox
    t, r    = solve_rk4_springBox(ic, W, tauE, tauI, T, dt, a, b, c, bias)
    end_idx = first_entry_and_stay_time(r, target_fp, threshold)

    if end_idx is None:
        r_keep  = r
        reached = False
    else:
        r_keep  = r[:end_idx + 1]
        reached = True

    if len(r_keep) < min_raw_length:
        return None, None

    r_keep = r_keep[:max_raw_length]
    meta   = {
        'label':              label,
        'ic_index':           idx_ic,
        'raw_length':         len(r),
        'kept_length':        len(r_keep),
        'reached_threshold':  reached,
    }
    return r_keep, meta


# =====================================================================
# Orbit block construction
# =====================================================================
def build_orbit_blocks_for_round(round_indices, orbits_kept, Js, r_stars,
                                 biases_S, meta_S, W, tauE, tauI, T_round,
                                 dt, a, b, c, N, scale_ic=1.0,
                                 no_freqs=3, wash_round=2.5):
    """
    Precompute orbit blocks for a set of system indices.

    Each system i in round_indices has an oscillatory IC determined by
    the IC index stored in orbits_kept[i]. The orbit is simulated via
    precompute_orbit and the steady-state portion is extracted.

    Parameters
    ----------
    round_indices : list of int — system indices (keys into orbits_kept)
    orbits_kept   : dict i -> (k_ic, label, r)
    Js            : np.ndarray, shape (N, N, n_systems) — Jacobians
    r_stars       : np.ndarray, shape (N, n_systems) — fixed points
    biases_S      : np.ndarray, shape (N, n_systems)
    meta_S        : list of dicts
    T_round       : float — simulation duration
    wash_round    : float — wash-in time (passed to precompute_orbit)

    Returns
    -------
    blocks : list of np.ndarray, each shape (T_i, N)
    labels : list of str
    meta   : list of dicts
    """
    from utils import init_cond
    from fig7.new_orbits_functions import precompute_all_orbits

    K          = len(round_indices)
    ic_orbits  = np.zeros((N, K))
    r_stars_r  = np.zeros((N, K))
    biases_r   = np.zeros((N, K))

    for col, i in enumerate(round_indices):
        k_ic, _, _ = orbits_kept[i]
        ICs                = init_cond(Js[:, :, i])
        ic_orbits[:, col]  = scale_ic * ICs[:, k_ic] + r_stars[:, i]
        r_stars_r[:, col]  = r_stars[:, i]
        biases_r[:, col]   = biases_S[:, i]

    infos = precompute_all_orbits(
        ic_orbits, W, tauE, tauI, T_round, dt, a, b, c,
        r_stars_r, biases_r, no_freqs=no_freqs, wash=wash_round)

    blocks, labels, meta = [], [], []
    for col, i in enumerate(round_indices):
        r_orb = infos[col]['r_orbit']
        if r_orb.shape[0] == N:
            r_orb = r_orb.T           # ensure (T, N)
        m = meta_S[i]
        blocks.append(r_orb)
        labels.append(f"sys{i}_{m['family']}{m['param']}")
        meta.append({
            'context':        i,
            'type':           'orbit',
            'system_index':   i,
            'family':         m['family'],
            'param_name':     m['param_name'],
            'param':          m['param'],
            'classification': orbits_kept[i][1],
            'dataset':        m.get('dataset', 1),
            'T':              T_round,
            'wash':           wash_round,
        })
    return blocks, labels, meta


# =====================================================================
# Greedy transient selection (maximize pooled dimensionality)
# =====================================================================
def _normalize_block_rms(block):
    """RMS-normalize a block (matches run_pca_trajectory_normalized)."""
    bc  = block - block.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(bc**2))
    return bc / rms if rms > 0 else bc


def _block_cov(block):
    """Return (X^T X, T) for a centered, RMS-normalized block."""
    bc = _normalize_block_rms(block)
    return bc.T @ bc, bc.shape[0]


def _n90_from_cov(C):
    """Compute n90 from a pooled covariance matrix via eigendecomposition."""
    w   = np.sort(np.linalg.eigvalsh(C))[::-1]
    cev = np.cumsum(w) / np.sum(w)
    return int(np.searchsorted(cev, 0.90) + 1), cev


def greedy_select_transients(transient_pool, orbit_blocks_50, n_pick=50):
    """
    Greedily select n_pick transients from transient_pool that maximally
    increase pooled dimensionality (n90) when combined with the 50 orbit blocks.

    At each step, the transient whose addition to the current pooled
    covariance maximally increases n90 is selected.

    Parameters
    ----------
    transient_pool   : list of np.ndarray, each shape (T_i, N)
    orbit_blocks_50  : list of 50 orbit arrays, each shape (T_k, N)
    n_pick           : int — how many transients to select

    Returns
    -------
    picked_indices : list of int — indices into transient_pool
    n90_history    : list of int — n90 at each step (length n_pick + 1)
    """
    N = orbit_blocks_50[0].shape[1]

    # Build seed covariance from orbits
    C_seed = np.zeros((N, N))
    for b in orbit_blocks_50:
        Cb, _ = _block_cov(b)
        C_seed += Cb

    # Precompute transient covariances
    trans_covs = np.array([_block_cov(b)[0] for b in transient_pool])

    remaining    = set(range(len(transient_pool)))
    picked       = []
    C_current    = C_seed.copy()
    n90_history  = [_n90_from_cov(C_current)[0]]

    for _ in tqdm(range(n_pick), desc='Greedy transient selection'):
        cand     = np.array(sorted(remaining))
        best_n90 = -1
        best_idx = None
        for i in cand:
            n90_try, _ = _n90_from_cov(C_current + trans_covs[i])
            if n90_try > best_n90:
                best_n90, best_idx = n90_try, i
        picked.append(int(best_idx))
        C_current = C_current + trans_covs[best_idx]
        remaining.remove(best_idx)
        n90_history.append(best_n90)

    return picked, n90_history


# =====================================================================
# Cross-projection: Elsayed et al. 2016
# =====================================================================
def compute_all_cross_projections_elsayed(norm_blocks, k=3):
    """
    Compute pairwise cross-projection matrix (Elsayed et al. 2016).

    For each pair (i, j):
        score(i -> j) = ||X_j @ V_i @ V_i^T||_F^2 / ||X_j_best_k||_F^2

    where V_i are the top-k right singular vectors of block i, and
    X_j_best_k is the projection of block j onto its own top-k subspace.

    This measures the fraction of block j's variance explained by
    block i's k-dimensional subspace.

    Parameters
    ----------
    norm_blocks : list of np.ndarray, each shape (T_i, N) — already normalized
    k           : int — number of PCs defining each block's subspace

    Returns
    -------
    cross_proj_matrix : np.ndarray, shape (n, n)
        cross_proj_matrix[i, j] = overlap of block j onto block i's subspace
        Diagonal is 1 by definition.
    """
    n       = len(norm_blocks)
    bases   = []   # V_i: top-k right singular vectors, shape (N, k)
    centered = []
    var_best_k = []

    for block in norm_blocks:
        bc = block - block.mean(axis=0, keepdims=True)
        centered.append(bc)
        _, S, Vt = np.linalg.svd(bc, full_matrices=False)
        bases.append(Vt[:k].T)          # (N, k)
        var_best_k.append(np.sum(S[:k]**2))

    cross_proj_matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc='Elsayed cross-projection'):
        for j in range(n):
            if i == j:
                cross_proj_matrix[i, j] = 1.0
            else:
                # Project block j onto block i's subspace
                Xt_proj = centered[j] @ bases[i] @ bases[i].T
                cross_proj_matrix[i, j] = np.sum(Xt_proj**2) / var_best_k[j]

    return cross_proj_matrix


# =====================================================================
# Sorting helpers for cross-projection matrix
# =====================================================================
def sort_by_total_overlap(M_block):
    """
    Sort block indices ascending by mean off-diagonal overlap
    (lowest overlap first = most distinct trajectories first).

    Parameters
    ----------
    M_block : np.ndarray, shape (n, n) — cross-projection submatrix

    Returns
    -------
    order : np.ndarray of int
    """
    M = M_block.copy()
    np.fill_diagonal(M, np.nan)
    mean_overlap = np.nanmean(M, axis=1)
    return np.argsort(mean_overlap)


def farthest_point_order(D):
    """
    Greedy farthest-point traversal on a dissimilarity matrix D.

    Starts from the most distinctive item (highest mean distance),
    then repeatedly picks the item farthest from anything already selected.

    Parameters
    ----------
    D : np.ndarray, shape (n, n) — dissimilarity matrix (D = 1 - similarity)

    Returns
    -------
    order : list of int
    """
    n    = D.shape[0]
    seed = int(np.argmax(D.mean(axis=1)))
    order = [seed]
    min_d = D[seed].copy()
    min_d[seed] = -np.inf
    for _ in range(n - 1):
        nxt = int(np.argmax(min_d))
        order.append(nxt)
        min_d = np.minimum(min_d, D[nxt])
        min_d[nxt] = -np.inf
    return order


# =====================================================================
# Tangling: Russo & Churchland 2018
# =====================================================================
def _forward_diff_per_condition(X, cond, sample_interval):
    """
    Forward finite difference of X within each condition, padded to keep
    the same number of rows. The last difference in each condition is
    repeated to fill.

    Parameters
    ----------
    X             : np.ndarray, shape (T, K)
    cond          : np.ndarray, shape (T,) — integer condition labels
    sample_interval : float — dt in seconds

    Returns
    -------
    Xdot : np.ndarray, shape (T, K)
    """
    T, K  = X.shape
    Xdot  = np.zeros_like(X)
    for c in np.unique(cond):
        idx = np.nonzero(cond == c)[0]
        if idx.size == 0:
            continue
        seg  = X[idx, :]
        if seg.shape[0] == 1:
            dpad = np.zeros_like(seg)
        else:
            d    = np.diff(seg, axis=0)          # (Tc-1, K)
            dpad = np.vstack([d, d[-1:, :]])     # pad last row
        Xdot[idx, :] = dpad / sample_interval
    return Xdot


def tangling_Q_cross_only(X, cond, sample_interval,
                           alpha=0.1, time_step=1, return_matches=True):
    """
    Tangling index Q(t) as defined in Russo & Churchland 2018.

    Q(t) = max_{t': cond[t'] != cond[t]}
               ||x_dot(t) - x_dot(t')||^2 / (||x(t) - x(t')||^2 + epsilon)

    where epsilon = alpha * sum_k Var(X_k, ddof=1).

    Only cross-condition pairs are compared, so trajectories of the same
    condition do not contribute (avoiding trivially high tangling at
    crossing points within a single trajectory).

    Parameters
    ----------
    X               : np.ndarray, shape (T, K)
    cond            : np.ndarray, shape (T,) — integer condition labels
    sample_interval : float — dt in seconds
    alpha           : float — epsilon scaling (default 0.1)
    time_step       : int — subsample stride (default 1 = all time points)
    return_matches  : bool — whether to store matched trajectory info

    Returns
    -------
    Q   : np.ndarray, shape (M,) — NaN where no cross-condition partner exists
    out : dict — diagnostic info (X, X_dot, times, matches if requested)
    """
    T, K  = X.shape
    ddof  = 1 if T > 1 else 0
    epsilon = alpha * np.sum(np.var(X, axis=0, ddof=ddof))

    Xdot      = _forward_diff_per_condition(X, cond, sample_interval)
    t_indices = np.arange(0, T, int(time_step))
    M         = t_indices.size

    Q  = np.full(M, np.nan)
    t2 = np.full(M, -1, dtype=int)
    if return_matches:
        y_sel  = np.full((M, K), np.nan)
        dy_sel = np.full((M, K), np.nan)

    for i, t in enumerate(t_indices):
        cand = t_indices[cond[t_indices] != cond[t]]
        if cand.size == 0:
            continue
        num   = np.sum((Xdot[t] - Xdot[cand])**2, axis=1)
        den   = np.sum((X[t]    - X[cand]   )**2, axis=1) + epsilon
        ratio = num / den
        j     = int(np.argmax(ratio))
        t2[i] = int(cand[j])
        Q[i]  = ratio[j]
        if return_matches:
            y_sel[i]  = X[cand[j]]
            dy_sel[i] = Xdot[cand[j]]

    out = {'X': X, 'X_dot': Xdot, 'conditionMask': cond,
           'times_t1': t_indices, 'times_t2': t2}
    if return_matches:
        out['y_sel']  = y_sel
        out['dy_sel'] = dy_sel

    return Q, out


def tangling_for_block_list(blocks, sample_interval=0.005,
                             alpha=0.1, time_step=1):
    """
    Pool a list of trajectory blocks, assign each block its own condition
    label, and compute cross-trajectory tangling (Russo & Churchland 2018).

    Parameters
    ----------
    blocks          : list of np.ndarray, each shape (T_i, K)
    sample_interval : float — dt in seconds

    Returns
    -------
    Q_clean : np.ndarray — non-NaN tangling values across all time points
    """
    X_pool   = np.vstack(blocks)
    traj_ids = np.concatenate([np.full(len(b), i) for i, b in enumerate(blocks)])
    Q, _     = tangling_Q_cross_only(X_pool, traj_ids,
                                      sample_interval=sample_interval,
                                      alpha=alpha, time_step=time_step)
    return Q[~np.isnan(Q)]
