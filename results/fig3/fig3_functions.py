import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# =====================================================================
# Eigenvalue collection
# =====================================================================
def collect_eigs(solutions_list, max_points=None, rng=None):
    """
    Concatenate all eigenvalues from a list of solution dicts,
    optionally subsampling to max_points.

    Parameters
    ----------
    solutions_list : list of dict, each with key 'Lambda'
    max_points     : int or None
    rng            : np.random.Generator or None

    Returns
    -------
    eigs : np.ndarray, complex, shape (M,)
    """
    eigs = np.concatenate([np.asarray(d['Lambda']).ravel() for d in solutions_list])
    if max_points is not None and eigs.size > max_points:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(eigs.size, size=max_points, replace=False)
        eigs = eigs[idx]
    return eigs


# =====================================================================
# Parallelized proxy amplification computation
# =====================================================================
def run_proxy_only_joblib(J_lists, dt=0.1, T=100, tauE=1.0, tauI=1.0,
                          x=None, n_jobs=-1, backend='loky', verbose=0):
    """
    Compute proxy_ampl for multiple lists of matrices in parallel.

    Parameters
    ----------
    J_lists : list of list of np.ndarray
        Each inner list is one ensemble of matrices.
    x       : np.ndarray or None
        Fixed initial condition. If None, uses init_cond(J)[:,0] per matrix.
    n_jobs  : int — joblib parallelism (-1 = all cores)

    Returns
    -------
    proxies : list of np.ndarray, one per inner list, each shape (n_mats,)
    """
    from utils import J_dynamics_numerical

    def _proxy_for_one_J(J):
        x_run = None if x is None else np.array(x, dtype=float, copy=True)
        _, proxy = J_dynamics_numerical(J, dt=dt, T=T, tauE=tauE, tauI=tauI, x=x_run)
        return float(proxy)

    proxies = []
    for J_list in J_lists:
        proxy_list = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(_proxy_for_one_J)(J) for J in J_list
        )
        proxies.append(np.asarray(proxy_list, dtype=float))
    return proxies


# =====================================================================
# Violin plot helper
# =====================================================================
def plot_half_violin(data_left, data_right,
                     colors_left=None, colors_right=None,
                     broken_axis=False, log_scale=False,
                     yticks=None, ylim=None,
                     bot_ylim=(-0.01, 0.01),
                     figsize=(1, 1), dpi=500,
                     width=0.7, height_ratios=(15, 1)):
    """
    Split violin plot: left half = non-normal, right half = normal.

    If broken_axis=True, the right half is shown as a flat line at y=0
    on a tiny bottom axis (for data where normal baseline = 0).
    If broken_axis=False, both halves are full violins on the same axis.

    Parameters
    ----------
    data_left   : list of array-like — one entry per position (reversed sigma order)
    data_right  : list of array-like — same length as data_left
    colors_left : list of str or None (defaults to blue palette)
    colors_right: list of str or None (defaults to gray)
    broken_axis : bool
    log_scale   : bool — apply log scale to main axis
    yticks      : list or None
    ylim        : [ymin, ymax] or None
    bot_ylim    : [ymin, ymax] for the bottom axis (broken_axis only)
    height_ratios: tuple — (top, bottom) height ratio (broken_axis only)
    """
    if colors_left is None:
        colors_left = ['#a8c6de', '#4682B4', '#2a4f6c']
    if colors_right is None:
        colors_right = ['gray'] * len(data_left)

    positions = np.arange(1, len(data_left) + 1)

    if broken_axis:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True, figsize=figsize, dpi=dpi,
            gridspec_kw={'height_ratios': list(height_ratios), 'hspace': 0.0}
        )
        ax = ax_top
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # --- left half violins ---
    parts_L = ax.violinplot(data_left, positions=positions, widths=width,
                            showmeans=False, showmedians=False, showextrema=False)
    for pos, (body, col) in enumerate(zip(parts_L['bodies'], colors_left), start=1):
        body.set_facecolor(col)
        body.set_edgecolor('black')
        body.set_alpha(1)
        body.set_linewidths(0.5)
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.minimum(verts[:, 0], pos)
        body.get_paths()[0].vertices = verts

    if broken_axis:
        # right half = flat line at y=0 on bottom axis
        ax_bot.set_ylim(bot_ylim)
        ax_bot.set_yticks([0])
        ax_bot.set_yticklabels([])
        halfwidth = width / 2
        for pos in positions:
            ax_bot.hlines(0, xmin=pos, xmax=pos + halfwidth, color='black', linewidth=0.6)

        # shared cosmetics
        for a in (ax_top, ax_bot):
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.set_xlim(0.4, 3.5)
        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(axis='x', bottom=False)
        ax_bot.tick_params(axis='x', top=False)
        ax_top.spines['left'].set_linewidth(0.5)
        ax_bot.spines['left'].set_linewidth(0.5)
        ax_bot.spines['bottom'].set_linewidth(0.5)
        ax_top.tick_params(axis='y', width=0.5, length=2)
        ax_bot.tick_params(axis='y', width=0.5, length=2)
        ax_bot.tick_params(axis='x', width=0.5, length=2)
        ax_bot.set_xticks(positions)
        ax_bot.set_xticklabels([])
        fig.subplots_adjust(hspace=0.0)

    else:
        # --- right half violins ---
        parts_R = ax.violinplot(data_right, positions=positions, widths=width,
                                showmeans=False, showmedians=False, showextrema=False)
        for pos, (body, col) in zip(positions, zip(parts_R['bodies'], colors_right)):
            body.set_facecolor(col)
            body.set_edgecolor('black')
            body.set_alpha(1)
            body.set_linewidth(0.5)
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.maximum(verts[:, 0], pos)
            body.get_paths()[0].vertices = verts

        # dashed center dividers
        for pos in positions:
            ax.vlines(pos, *(ylim if ylim else ax.get_ylim()),
                      colors='black', linestyles=(0, (2, 2)),
                      linewidth=0.4, alpha=1, zorder=3)

        ax.set_xlim(0.4, len(positions) + 0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(axis='x', width=0.5, length=2)
        ax.tick_params(axis='y', width=0.5, length=2)

    if log_scale:
        ax.set_yscale('log')
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.show()
    
def compute_fractions(data):
    """
    Compute per-matrix fractions of each behavioural class.

    Parameters
    ----------
    data : list of length n_matrices
        Each element is a list of label strings for IC outcomes.

    Returns
    -------
    p_osc, p_base, p_newfp, p_chaos : np.ndarray, shape (n_matrices,)
    """
    label_to_class = {
        'periodic': 'osc', 'quasi-periodic': 'osc', 'quasiperiodic': 'osc',
        '0_fp': 'baseline', 'return_to_baseline': 'baseline', '0_fixed_point': 'baseline',
        'new_fixed_point': 'new_fp', 'new_fp': 'new_fp',
        'chaotic': 'chaos', 'chaos': 'chaos', 'uncertain': 'chaos'
    }
    class_names = ['osc', 'baseline', 'new_fp', 'chaos']
    n_ics = len(data[0])
    frac = np.zeros((len(data), 4), dtype=float)
    for i, row in enumerate(data):
        for lab in row:
            frac[i, class_names.index(label_to_class[lab])] += 1.0
        frac[i] /= n_ics
    return frac[:, 0], frac[:, 1], frac[:, 2], frac[:, 3]


def plot_osc_fractions(p_osc_non_normal, p_osc_normal, eps=0.0,
                       figsize=(1, 1), dpi=500):
    """
    Bar plot of fraction of matrices showing oscillations,
    split by non-normal (colored) vs normal (black).

    Parameters
    ----------
    p_osc_non_normal : list of 3 arrays — [sigma_large, sigma_med, sigma_small]
    p_osc_normal     : list of 3 arrays — same order
    eps              : float — threshold above which a matrix 'has' oscillations
    """
    n = len(p_osc_non_normal)
    frac_nn = np.array([np.sum(p > eps) / len(p) for p in p_osc_non_normal])
    frac_n  = np.array([np.sum(p > eps) / len(p) for p in p_osc_normal])

    x = np.arange(n)
    width, spacing = 0.28, 0.20
    colors_nn = ['#a8c6de', '#4682B4', '#2a4f6c']

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars1 = ax.bar(x - (width/2 + spacing/2), frac_nn, width,
                   color='white', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + (width/2 + spacing/2), frac_n,  width,
                   color='black', edgecolor='black', linewidth=0.5)
    for bar, col in zip(bars1, colors_nn):
        bar.set_facecolor(col)

    ax.set_xticks(x);        ax.set_xticklabels(['', '', ''])
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(['', '', ''])
    ax.set_ylim(-0.03, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.45)
    ax.spines['bottom'].set_linewidth(0.45)
    ax.tick_params(length=2, width=0.45)
    plt.tight_layout()
    plt.show()
    
def ternary_barycentric_to_cartesian(b1, b2, b3):
    """Convert barycentric ternary coordinates to 2D Cartesian."""
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3)/2])
    x = b1*v1[0] + b2*v2[0] + b3*v3[0]
    y = b1*v1[1] + b2*v2[1] + b3*v3[1]
    return x, y


def _ternary_setup(ax, border_lw=0.1):
    """Draw triangle border and strip axes."""
    h = np.sqrt(3) / 2
    ax.plot([0.0, 1.0, 0.5, 0.0], [0.0, 0.0, h, 0.0],
            color='k', linewidth=border_lw, zorder=0)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])


def plot_ternary_main_with_cbar_and_baseline_pdf(
        p_osc, p_base, p_newfp, p_chaos,
        eps=0.0, vmin=0.0, vmax=1.0,
        cbar_shrink=0.88, cbar_thickness=0.55,
        pdf_height=0.06, gap=0.0015,
        cbar_outline_lw=0.3, cbar_outline_color='k',
        pdf_outline_lw=0.3, pdf_outline_color='k',
        pdf_facecolor='white', figsize=(1, 1), dpi=500):
    """
    Ternary scatter coloured by baseline fraction, with horizontal colorbar
    and baseline KDE below it.

    Parameters
    ----------
    p_osc, p_base, p_newfp, p_chaos : np.ndarray, shape (n_matrices,)
    eps          : float — minimum non-baseline fraction to include a matrix
    vmin, vmax   : float — colorbar range
    cbar_shrink  : float — colorbar width relative to main axis
    cbar_thickness: float — relative thickness of colorbar strip
    pdf_height   : float — height of KDE axis in figure coords
    gap          : float — gap between colorbar and KDE in figure coords
    """
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    p_non = 1.0 - p_base
    mask  = p_non > eps
    if mask.sum() == 0:
        print('No matrices with non-baseline activity > eps.')
        return

    p_osc_m, p_chaos_m = p_osc[mask], p_chaos[mask]
    p_newfp_m, p_base_m, p_non_m = p_newfp[mask], p_base[mask], p_non[mask]

    q_osc   = p_osc_m   / p_non_m
    q_chaos = p_chaos_m / p_non_m
    q_newfp = p_newfp_m / p_non_m
    s = q_osc + q_chaos + q_newfp
    q_osc /= s; q_chaos /= s; q_newfp /= s

    x, y = ternary_barycentric_to_cartesian(q_osc, q_chaos, q_newfp)
    sort_idx = np.argsort(p_base_m)
    x_s, y_s, pb_s = x[sort_idx], y[sort_idx], p_base_m[sort_idx]

    cmap_full = plt.get_cmap('plasma')
    cmap_mid  = LinearSegmentedColormap.from_list(
        'plasma_mid', cmap_full(np.linspace(0, 0.8, 256)))
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax  = fig.add_axes([0.08, 0.17, 0.84, 0.78])
    _ternary_setup(ax)
    ax.scatter(x_s, y_s, c=pb_s, cmap=cmap_mid, norm=norm,
               s=2, alpha=0.5, edgecolors='k', linewidths=0.05, zorder=1)

    # colorbar
    main_pos = ax.get_position()
    bar_w  = cbar_shrink * main_pos.width
    bar_x0 = main_pos.x0 + 0.5 * (main_pos.width - bar_w)
    bar_h  = 0.035 * cbar_thickness
    bar_y0 = 0.15
    pdf_y0 = bar_y0 - gap - pdf_height

    ax_cbar = fig.add_axes([bar_x0, bar_y0, bar_w, bar_h])
    ax_pdf  = fig.add_axes([bar_x0, pdf_y0, bar_w, pdf_height], sharex=ax_cbar)

    sm = ScalarMappable(norm=norm, cmap=cmap_mid)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
    cb.solids.set_alpha(0.5); cb.solids.set_edgecolor('face'); cb.solids.set_linewidth(0)
    cb.set_ticks([]); cb.outline.set_visible(False)
    ax_cbar.set_yticks([]); ax_cbar.tick_params(bottom=False, labelbottom=False)
    ax_cbar.set_frame_on(True)
    for side in ['top', 'bottom', 'left', 'right']:
        sp = ax_cbar.spines[side]
        sp.set_visible(True); sp.set_linewidth(cbar_outline_lw); sp.set_edgecolor(cbar_outline_color)

    # baseline KDE
    kde = gaussian_kde(p_base_m)
    xx  = np.linspace(vmin, vmax, 2000)
    yy  = kde(xx); yk = -yy

    ax_pdf.set_facecolor(pdf_facecolor); ax_pdf.set_frame_on(True)
    for side in ['top', 'bottom', 'left', 'right']:
        ax_pdf.spines[side].set_visible(False)
    ax_pdf.spines['top'].set_visible(True)
    ax_pdf.spines['top'].set_linewidth(pdf_outline_lw)
    ax_pdf.spines['top'].set_edgecolor(pdf_outline_color)
    ax_pdf.set_xticks([]); ax_pdf.set_yticks([])

    ax_pdf.fill_between(xx, 0, yk, color='#2a4f6c', alpha=1, linewidth=0,
                        edgecolor='none', antialiased=False)
    x_poly = np.concatenate(([xx[0]], xx, [xx[-1]]))
    y_poly = np.concatenate(([0.0],   yk, [0.0]))
    ax_pdf.plot(x_poly, y_poly, color='k', linewidth=0.3,
                solid_joinstyle='round', solid_capstyle='round', antialiased=True)
    ax_pdf.plot([xx[0],  xx[0]],  [0.0, yk[0]],  color='k', linewidth=0.6)
    ax_pdf.plot([xx[-1], xx[-1]], [0.0, yk[-1]], color='k', linewidth=0.6)
    ax_pdf.set_xlim(vmin, vmax)
    ax_pdf.set_ylim(-yy.max() * 1.05, 0.0)

    plt.show()
    return fig, (ax, ax_cbar, ax_pdf)


def plot_ternary_kde_only(p_osc, p_base, p_newfp, p_chaos,
                          eps=0.0, figsize=(1, 1), dpi=500):
    """
    Separate figure: KDE density contourf on the ternary triangle.

    Parameters
    ----------
    p_osc, p_base, p_newfp, p_chaos : np.ndarray, shape (n_matrices,)
    eps : float — minimum non-baseline fraction to include a matrix
    """
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap

    p_non = 1.0 - p_base
    mask  = p_non > eps
    if mask.sum() == 0:
        print('No matrices with non-baseline activity > eps.')
        return

    q_osc   = p_osc[mask]   / p_non[mask]
    q_chaos = p_chaos[mask] / p_non[mask]
    q_newfp = p_newfp[mask] / p_non[mask]
    s = q_osc + q_chaos + q_newfp
    q_osc /= s; q_chaos /= s; q_newfp /= s

    x, y = ternary_barycentric_to_cartesian(q_osc, q_chaos, q_newfp)
    kde   = gaussian_kde(np.vstack([x, y]))

    h = np.sqrt(3) / 2
    grid_x, grid_y = np.mgrid[0:1:200j, 0:h:200j]
    z = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    mask_tri = ((grid_y >= 0) &
                (grid_y <= np.sqrt(3) * grid_x) &
                (grid_y <= -np.sqrt(3) * grid_x + np.sqrt(3)))
    z_masked = np.where(mask_tri, z, np.nan)

    cmap_blue = LinearSegmentedColormap.from_list(
        'blue_gray_r', ['#ffffff', '#2a4f6c', '#000000'], N=256)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _ternary_setup(ax, border_lw=0.1)
    ax.contourf(grid_x, grid_y, z_masked, levels=15,
                cmap=cmap_blue, alpha=1, zorder=0)
    plt.show()
    return fig
