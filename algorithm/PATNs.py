
import numpy as np
import scipy.linalg
from scipy.optimize import linprog
from scipy.stats import truncnorm
from itertools import combinations, product
from matplotlib import pyplot as plt


# =====================================================================
# Orthogonal-matrix helper
# =====================================================================
def ortho_first_vector(v, rng):
    d = len(v)
    if d == 1:
        return np.ones((1, 1))
    v = v / np.linalg.norm(v)
    A = rng.standard_normal((d, d))
    A[:, 0] = v
    Q, R = np.linalg.qr(A)
    if np.dot(Q[:, 0], v) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# =====================================================================
# Turn block_sizes into a plain list and check it sums to N.
# =====================================================================
def make_size_list(N, block_sizes):
    if np.isscalar(block_sizes):
        size_list = [int(block_sizes)] * (N // int(block_sizes))
    else:
        size_list = list(block_sizes)
    assert sum(size_list) == N, f"sizes sum to {sum(size_list)}, not N={N}"
    return size_list


# =====================================================================
# Single source of truth for the column order. One entry per column,
# as (kind, block):
#   ('U', -1)  the uniform column (always first)
#   ('C', m)   the coarse column belonging to block m   (m = 1 .. P-1)
#   ('L', m)   a localized column of block m
#
# Block 0 has no coarse column (the uniform absorbs its slot), so there
# are P-1 coarse columns in total.
# =====================================================================
def make_layout(size_list, coarse_placement):
    P = len(size_list)
    layout = []
    layout.append(('U', -1))
    if coarse_placement == 'beginning':
        for m in range(1, P):
            layout.append(('C', m))
        for m in range(P):
            for _ in range(size_list[m] - 1):
                layout.append(('L', m))
    elif coarse_placement == 'end':
        for m in range(P):
            for _ in range(size_list[m] - 1):
                layout.append(('L', m))
        for m in range(1, P):
            layout.append(('C', m))
    elif coarse_placement == 'per_block':
        for m in range(P):
            if m >= 1:
                layout.append(('C', m))
            for _ in range(size_list[m] - 1):
                layout.append(('L', m))
    else:
        raise ValueError("coarse_placement must be 'beginning', 'end', or 'per_block'")
    return layout


# =====================================================================
# Draw the target eigenvalues and build the real Schur diagonal matrix.
#
#   lam0              : value of the uniform mode (real, settable)
#   coarse_vals       : length P-1, value of each coarse mode (real, settable)
#   n_real_per_block  : length P, how many localized eigenvalues are real
#                       in each block; the rest become complex pairs.
#                       Requires (size-1 - n_real) to be even and >= 0.
#   real_position     : 'beginning' or 'end' of each block's localized run.
#   real eigenvalues  : drawn from the same truncated normal as the real
#                       parts of the complex ones (centered mu_r, capped mu_r).
# =====================================================================
def sample_eigenvalues(N, size_list, n_real_per_block, coarse_vals, lam0=0.0,
                       real_position='end', coarse_placement='beginning',
                       mu_r=0.7, sigma_r=1, mu_im=0.0, sigma_im=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    P = len(size_list)
    assert len(n_real_per_block) == P, "n_real_per_block must have length P"
    assert len(coarse_vals) == P - 1, "coarse_vals must have length P-1"

    n_pairs = [0] * P
    for m in range(P):
        loc = size_list[m] - 1
        r = n_real_per_block[m]
        assert 0 <= r <= loc, "n_real out of range in block %d" % m
        assert (loc - r) % 2 == 0, "block %d: (size-1 - n_real) must be even" % m
        n_pairs[m] = (loc - r) // 2

    total_pairs = sum(n_pairs)
    total_reals = sum(n_real_per_block)

    re_pair = truncnorm((-np.inf - mu_r) / sigma_r, 0, loc=mu_r, scale=sigma_r).rvs(total_pairs, random_state=rng)
    im_pair = truncnorm((-np.inf - mu_im) / sigma_im, 0, loc=mu_im, scale=sigma_im).rvs(total_pairs, random_state=rng)
    if total_reals > 0:
        re_real = truncnorm((-np.inf - mu_r) / sigma_r, 0, loc=mu_r, scale=sigma_r).rvs(total_reals, random_state=rng)
    else:
        re_real = np.array([])

    loc_eigs = [[] for _ in range(P)]
    pair_idx = 0
    real_idx = 0
    for m in range(P):
        pairs_list = []
        for _ in range(n_pairs[m]):
            val = re_pair[pair_idx] + 1j * im_pair[pair_idx]
            pair_idx += 1
            pairs_list.append(val)
            pairs_list.append(np.conj(val))
        reals_list = []
        for _ in range(n_real_per_block[m]):
            reals_list.append(re_real[real_idx] + 0j)
            real_idx += 1
        if real_position == 'beginning':
            loc_eigs[m] = reals_list + pairs_list
        else:
            loc_eigs[m] = pairs_list + reals_list

    layout = make_layout(size_list, coarse_placement)
    Lambda = []
    loc_used = [0] * P
    for pos in range(len(layout)):
        kind = layout[pos][0]
        m = layout[pos][1]
        if kind == 'U':
            Lambda.append(lam0 + 0j)
        elif kind == 'C':
            Lambda.append(coarse_vals[m - 1] + 0j)
        else:
            Lambda.append(loc_eigs[m][loc_used[m]])
            loc_used[m] += 1
    Lambda = np.array(Lambda)

    Lambda_Schur, _ = scipy.linalg.cdf2rdf(Lambda, np.ones((N, N)))
    return Lambda, Lambda_Schur


# =====================================================================
# Build the orthogonal Q. Columns follow the same make_layout order,
# so they line up one-to-one with the eigenvalues above.
# =====================================================================
def sample_Q(N, blocks, coarse_placement='beginning', seed=None):
    rng = np.random.default_rng(seed)
    P = len(blocks)
    sizes = [len(b) for b in blocks]

    F = np.zeros((N, P))
    loc_cols = [[] for _ in range(P)]
    for m in range(P):
        idx = np.array(blocks[m])
        Qb = ortho_first_vector(np.ones(len(idx)), rng)
        for r in range(len(idx)):
            F[idx[r], m] = Qb[r, 0]
        for k in range(1, len(idx)):
            col = np.zeros(N)
            for r in range(len(idx)):
                col[idx[r]] = Qb[r, k]
            loc_cols[m].append(col)

    weights = np.sqrt(np.array(sizes) / N)
    G = F @ ortho_first_vector(weights, rng)

    layout = make_layout(sizes, coarse_placement)
    Q = np.zeros((N, N))
    loc_used = [0] * P
    for pos in range(len(layout)):
        kind = layout[pos][0]
        m = layout[pos][1]
        if kind == 'U':
            Q[:, pos] = G[:, 0]
        elif kind == 'C':
            Q[:, pos] = G[:, m]
        else:
            Q[:, pos] = loc_cols[m][loc_used[m]]
            loc_used[m] += 1
    return Q, dict(sizes=sizes, P=P, blocks=blocks, coarse_placement=coarse_placement)


# =====================================================================
# Scatter neurons into P blocks of the given sizes, spreading the
# inhibitory neurons proportionally (>=1 inhib and >=1 excit per block).
# =====================================================================
def random_block_assignment(N, block_sizes, inhib_idx, min_inhib=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sizes = make_size_list(N, block_sizes)
    P = len(sizes)

    inhib_set = set(int(i) for i in inhib_idx)
    inhib = np.array(sorted(inhib_set))
    excit = np.array([i for i in range(N) if i not in inhib_set])
    nI = len(inhib)
    assert nI >= P * min_inhib

    target = np.array([nI * sizes[m] / N for m in range(P)])
    k = np.floor(target).astype(int)
    while k.sum() < nI:
        k[np.argmax(target - k)] += 1
    for m in range(P):
        k[m] = np.clip(k[m], min_inhib, sizes[m] - 1)
    while k.sum() > nI:
        k[np.argmax(k - min_inhib)] -= 1
    while k.sum() < nI:
        k[np.argmax((np.array(sizes) - 1) - k)] += 1

    rng.shuffle(inhib)
    rng.shuffle(excit)
    blocks = []
    pi, pe = 0, 0
    for m in range(P):
        bI = inhib[pi:pi + k[m]]
        pi += k[m]
        bE = excit[pe:pe + sizes[m] - k[m]]
        pe += sizes[m] - k[m]
        blocks.append(np.sort(np.concatenate([bI, bE])))
    return blocks, k


def compute_M(N, Q):
    M = np.empty((N**2, int((N**2 - N) / 2)))
    prod = np.array(list(product(np.arange(0, N), repeat=2)))
    comb = np.array(list(combinations(np.arange(0, N), 2)))

    for i in range(N**2):
        M[i, :] = Q[prod[i, 0], comb[:, 0]] * Q[prod[i, 1], comb[:, 1]]
    return M


def columns_M_todelete(Lambda_Schur, N, M):
    off_diagonal_idx = np.triu_indices(N, 1)
    index = np.where(Lambda_Schur[off_diagonal_idx] != 0)
    rows = off_diagonal_idx[0][index]
    columns = off_diagonal_idx[1][index]

    idxs_M_delete = []
    for i in range(len(rows)):
        idx_M_delete = np.sum(np.arange(N - rows[i], N)) + columns[i] - rows[i] - 1
        idxs_M_delete.append(idx_M_delete)

    M = np.delete(M, idxs_M_delete, 1)
    return M, idxs_M_delete


def calculate_u(Lambda_Schur, Q, M_raw, N, frac_inhib, maxw, sizes,
                coarse_placement, coupling='between_via_global', objective='small_norm',
                l1_weights=(1.0, 1.0, 1.0),
                target_cross=None, random_cross=None,
                cross_reward=5.0, maxw_cross=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    valid = ('within', 'between_via_global', 'between_via_cross', 'between_via_global+cross')
    if coupling not in valid:
        raise ValueError("coupling must be one of " + str(valid))

    normal_part = Q @ Lambda_Schur @ Q.T
    w0 = normal_part.flatten()
    M_raw_reduced, idxs_todelete = columns_M_todelete(Lambda_Schur, N, M_raw)
    n = int(N**2)
    diagonal_indices = [i * (N + 1) for i in range(N)]
    inhib_indices = list(np.concatenate([np.arange(N * (i + 1) - frac_inhib, N * (i + 1)) for i in range(N)]))
    excit_indices = list(np.concatenate([np.arange(N * i, N * (i + 1) - frac_inhib) for i in range(N)]))

    M = M_raw_reduced.copy()
    normal_part_vector = -w0.copy()
    for i in range(N):
        M[N * i:N * (i + 1) - frac_inhib, :] *= -1
        normal_part_vector[N * i:N * (i + 1) - frac_inhib] *= -1

    p   = np.zeros(n); p[diagonal_indices] = 1
    exc = np.zeros(n); exc[excit_indices]  = 1
    inh = np.zeros(n); inh[inhib_indices]  = 1
    exc[diagonal_indices] = 0
    inh[diagonal_indices] = 0

    eps = 1e-6
    norm = lambda v: v / (np.linalg.norm(v, ord=np.inf) + eps)
    wd, we, wi = l1_weights
    c = -(wd * norm(M.T @ p) + we * norm(M.T @ exc) + wi * norm(M.T @ inh))
    if objective != 'small_norm':
        c = -c

    layout = make_layout(sizes, coarse_placement)
    col_block = np.full(N, -1, dtype=int)
    for pos in range(N):
        if layout[pos][0] == 'L':
            col_block[pos] = layout[pos][1]

    rows_idx, cols_idx = np.triu_indices(N, 1)
    kept = np.array(sorted(set(range(len(rows_idx))) - set(idxs_todelete)))
    s_blk = col_block[rows_idx[kept]]
    t_blk = col_block[cols_idx[kept]]
    both_local     = (s_blk != -1) & (t_blk != -1)
    touches_global = ~both_local
    within = both_local & (s_blk == t_blk)
    cross  = both_local & (s_blk != t_blk)

    allow_global = coupling in ('between_via_global', 'between_via_global+cross')
    allow_cross  = coupling in ('between_via_cross',  'between_via_global+cross')
    keep = within.copy()
    if allow_global:
        keep = keep | touches_global
    if allow_cross:
        keep = keep | cross

    is_target = np.zeros(len(kept), dtype=bool)
    if target_cross:
        pairs = {tuple(sorted(map(int, ab[:2]))) for ab in target_cross}
        for a, b in pairs:
            is_target |= cross & (((s_blk == a) & (t_blk == b)) | ((s_blk == b) & (t_blk == a)))
    elif random_cross:
        pool = np.where(cross)[0]
        k = min(int(random_cross), len(pool))
        is_target[rng.choice(pool, size=k, replace=False)] = True

    if is_target.any():
        keep = keep | is_target
        c = c.copy()
        c[is_target] -= cross_reward * rng.choice([-1.0, 1.0], size=int(is_target.sum()))

    mwc = maxw if maxw_cross is None else maxw_cross
    bounds = [((-mwc, mwc) if is_target[i] else (-maxw, maxw)) if keep[i] else (0.0, 0.0)
              for i in range(len(kept))]

    u = linprog(c=c, A_ub=M, b_ub=normal_part_vector, bounds=bounds,
                method='highs-ipm', options={'run_crossover': True})
    if not u.success:
        raise RuntimeError("LP failed: " + u.message)
    return u, idxs_todelete


def create_W(N, u_solution, idxs_todelete, Lambda_Schur, Q):
    inds = np.triu_indices(N, 1)
    big_u = np.zeros((N, N))
    maxSize = int((N**2 - N) / 2)

    u_triang = np.zeros(maxSize)
    filled_idx = sorted(set(range(maxSize)) - set(idxs_todelete))
    u_triang[filled_idx] = u_solution

    big_u[inds] = u_triang
    T = Lambda_Schur + big_u
    
    #Plot U
    fig, ax = plt.subplots(figsize=(1, 1), dpi=500)
    m = np.abs(big_u).max()
    ax.imshow(big_u, cmap='seismic',vmin=-m, vmax=m)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    W = Q @ T @ Q.T
    return W, big_u


def find_solution(N, block_sizes, maxw, etoi, n_real_per_block, coarse_vals,
                  lam0=0.0, real_position='beginning', coarse_placement='beginning',
                  l1_weights=(0, 0, 0),
                  coupling='between_via_global', target_cross=None, random_cross=None,
                  cross_reward=5.0, maxw_cross=None, seed=None):
    """
    Construct a Dale-compliant recurrent network with a prescribed eigenspectrum.

    Parameters
    ----------
    N : int
        Number of neurons.
    block_sizes : int or list of int
        Size of each block. If scalar, all blocks have equal size.
    maxw : float
        Maximum absolute weight in the LP bounds.
    etoi : float
        Excitatory-to-inhibitory ratio (e.g. 4 means 4:1, so 1/5 are inhibitory).
    n_real_per_block : list of int
        Number of real localized eigenvalues per block (rest form complex pairs).
    coarse_vals : list of float or complex
        Eigenvalues for the P-1 coarse modes.
    lam0 : float
        Eigenvalue of the global uniform mode.
    real_position : str
        'beginning' or 'end' — placement of real eigenvalues within each block's run.
    coarse_placement : str
        'beginning', 'end', or 'per_block' — column ordering in Q.
    l1_weights : tuple of 3 floats
        Relative weights (diagonal, excitatory off-diagonal, inhibitory off-diagonal)
        in the L1 objective.
    coupling : str
        One of 'within', 'between_via_global', 'between_via_cross',
        'between_via_global+cross'.
    target_cross : list of (int, int) or None
        Specific block pairs to target with cross couplings.
    random_cross : int or None
        Number of random cross couplings to add.
    cross_reward : float
        Reward weight for targeted cross couplings in the objective.
    maxw_cross : float or None
        Separate bound for targeted cross-coupling entries (defaults to maxw).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        W        : np.ndarray (N, N) — the connectivity matrix
        Q        : np.ndarray (N, N) — the orthogonal Schur basis
        Lambda   : np.ndarray (N,)  — the prescribed complex eigenvalues
        sol      : np.ndarray       — the LP solution vector
        info     : dict             — block structure metadata
        status   : str              — LP solver message
    """
    rng = np.random.default_rng(seed)
    frac_inhib = int(N / (etoi + 1))
    inhib_idx = np.arange(N - frac_inhib, N)

    blocks, _ = random_block_assignment(N, block_sizes, inhib_idx, rng=rng)
    Q, info = sample_Q(N, blocks, coarse_placement=coarse_placement, seed=seed)
    sizes = info['sizes']

    Lambda, Lambda_Schur = sample_eigenvalues(
        N, sizes, n_real_per_block, coarse_vals, lam0=lam0,
        real_position=real_position, coarse_placement=coarse_placement, rng=rng)

    M_raw = compute_M(N, Q)
    u, idxs_todelete = calculate_u(
        Lambda_Schur, Q, M_raw, N, frac_inhib, maxw,
        sizes, coarse_placement, coupling=coupling,
        l1_weights=l1_weights,
        target_cross=target_cross, random_cross=random_cross,
        cross_reward=cross_reward, maxw_cross=maxw_cross)

    print(u.message)
    W, big_u = create_W(N, u.x, idxs_todelete, Lambda_Schur, Q)
    return dict(W=W, Q=Q, Lambda=Lambda, sol=u.x, info=info, status=u.message, big_u=big_u)
