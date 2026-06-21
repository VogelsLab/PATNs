import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path
from scipy.linalg import svdvals


# =====================================================================
# Pseudospectrum computation
# =====================================================================
def compute_pseudospectrum_grid(J, grid_points=120, xy_lim=4):
    """
    Compute σ_min(zI - J) on a uniform grid in the complex plane.

    Parameters
    ----------
    J           : np.ndarray, shape (N, N)
    grid_points : int — grid resolution per axis
    xy_lim      : float — grid spans [-xy_lim, xy_lim] on both axes

    Returns
    -------
    X, Y      : np.ndarray, shape (grid_points, grid_points) — meshgrid
    sigma_min : np.ndarray, shape (grid_points, grid_points)
    """
    x = np.linspace(-xy_lim, xy_lim, grid_points)
    y = np.linspace(-xy_lim, xy_lim, grid_points)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    n = J.shape[0]
    I = np.eye(n, dtype=complex)
    sigma_min = np.empty_like(X, dtype=float)

    for i in range(grid_points):
        for j in range(grid_points):
            sigma_min[i, j] = svdvals(Z[i, j] * I - J)[-1]

    return X, Y, sigma_min


# =====================================================================
# Pseudospectrum plot
# =====================================================================
def plot_pseudospectrum(X, Y, sigma_min, eigs,
                        eps_levels=(1e-3, 1e-2, 1e-1, 1e0),
                        xy_lim=None, x_scale=4,
                        color_eigs='k', ax=None):
    """
    Plot pseudospectrum contour lines, colored blue left of Re=0 and red right.

    Shade of each segment depends on its distance from Re=0, normalized by
    x_scale (keep fixed across plots for color consistency).
    Disconnected contour pieces are not joined.

    Parameters
    ----------
    X, Y      : meshgrid arrays
    sigma_min : precomputed σ_min grid
    eigs      : complex eigenvalues to overlay
    eps_levels: contour levels (ε values)
    xy_lim    : axis limit (defaults to max of grid)
    x_scale   : normalization for color shading (fix across plots)
    color_eigs: color for eigenvalue scatter
    ax        : existing axes or None (creates new figure)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(1, 1), dpi=500)

    if xy_lim is None:
        xy_lim = max(np.max(np.abs(X)), np.max(np.abs(Y)))

    levels = sorted(eps_levels)

    cs = ax.contour(X, Y, sigma_min, levels=levels,
                    linewidths=0.6, colors='k')

    manual_positions = [
        (-xy_lim * 0.25,  0),
        (-xy_lim * 0.7,   0.1),
        ( xy_lim * 0.2,   0.1),
        ( xy_lim * 0.7,   0.1),
    ]
    ax.clabel(cs, inline=True, fontsize=3,
              fmt=lambda v: f'{v:g}', colors='black',
              manual=manual_positions)

    cmap_right = plt.get_cmap('Reds')
    cmap_left  = plt.get_cmap('GnBu')
    lw         = 0.5
    tmin, tmax = 0.2, 1.0

    def draw_chunk(chunk_verts):
        if len(chunk_verts) < 2:
            return
        verts = np.asarray(chunk_verts)
        x, y  = verts[:, 0], verts[:, 1]
        for i in range(len(verts) - 1):
            x1, y1 = x[i],   y[i]
            x2, y2 = x[i+1], y[i+1]
            x_mid = 0.5 * (x1 + x2)
            dist  = min(abs(x_mid) / x_scale, 1.0)
            t     = tmin + (tmax - tmin) * dist
            col_r = cmap_right(t)
            col_l = cmap_left(t)

            if x1 > 0 and x2 > 0:
                ax.plot([x1, x2], [y1, y2], color=col_r, linewidth=lw)
            elif x1 < 0 and x2 < 0:
                ax.plot([x1, x2], [y1, y2], color=col_l, linewidth=lw)
            elif x1 == 0 and x2 == 0:
                ax.plot([x1, x2], [y1, y2], color='0.5', linewidth=lw)
            else:
                if x2 != x1:
                    t_cross = max(0.0, min(1.0, -x1 / (x2 - x1)))
                    xc = 0.0
                    yc = y1 + t_cross * (y2 - y1)
                    if x1 < 0:
                        ax.plot([x1, xc], [y1, yc], color=col_l, linewidth=lw)
                        ax.plot([xc, x2], [yc, y2], color=col_r, linewidth=lw)
                    else:
                        ax.plot([x1, xc], [y1, yc], color=col_r, linewidth=lw)
                        ax.plot([xc, x2], [yc, y2], color=col_l, linewidth=lw)
                else:
                    ax.plot([x1, x2], [y1, y2], color='0.5', linewidth=lw)

    for coll in cs.collections:
        for path in coll.get_paths():
            verts = path.vertices
            codes = path.codes
            if codes is None:
                draw_chunk(verts)
            else:
                chunk = []
                for v, c in zip(verts, codes):
                    if c == Path.MOVETO:
                        if chunk:
                            draw_chunk(chunk)
                            chunk = []
                        chunk.append(v)
                    elif c == Path.LINETO:
                        chunk.append(v)
                    elif c == Path.CLOSEPOLY:
                        if chunk:
                            draw_chunk(chunk)
                            chunk = []
                    else:
                        if chunk:
                            draw_chunk(chunk)
                            chunk = []
                if chunk:
                    draw_chunk(chunk)

    for coll in cs.collections:
        coll.remove()

    eigs = np.asarray(eigs)
    ax.scatter(eigs.real, eigs.imag,
               edgecolor='k', color=color_eigs, s=1.5, linewidth=0.07)
    ax.axvline(0.0, color='gray', linestyle='--', linewidth=0.5)

    ax.set_xlim(-xy_lim, xy_lim); ax.set_ylim(-xy_lim, xy_lim)
    ax.set_xticks([-4, 0, 4]);    ax.set_xticklabels([])
    ax.set_yticks([-4, 0, 4]);    ax.set_yticklabels([])
    ax.tick_params(length=2, width=0.5)
    ax.spines['top'].set_visible(False);   ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5);  ax.spines['bottom'].set_linewidth(0.5)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    return ax


# =====================================================================
# Colorbar for the signed-distance coloring
# =====================================================================
def signed_distance_cmap(cmap_left='GnBu', cmap_right='Reds',
                          tmin=0.2, tmax=1.0, n=256):
    """
    Colormap from cmap_left (x < 0) to cmap_right (x > 0),
    with shade proportional to distance from Re = 0.
    """
    cmapL = plt.get_cmap(cmap_left)
    cmapR = plt.get_cmap(cmap_right)
    n2    = n // 2
    t     = np.linspace(tmax, tmin, n2)
    colors = np.vstack([cmapL(t), cmapR(t[::-1])])
    return LinearSegmentedColormap.from_list('signed_distance', colors)


def plot_signed_distance_colorbar(x_scale=4, figsize=(0.05, 1), dpi=500, lw=0.5):
    """Standalone colorbar for the pseudospectrum signed-distance coloring."""
    cmap = signed_distance_cmap()
    norm = Normalize(vmin=-x_scale, vmax=x_scale)
    sm   = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cbar = fig.colorbar(sm, cax=ax, orientation='vertical')
    cbar.outline.set_linewidth(lw)
    cbar.set_ticks([])
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(lw)
    plt.show()
    return fig, ax

# =====================================================================
# Noise robustness violin plots
# =====================================================================
def _left_half_violin(ax, data, positions, colors=None, widths=0.7):
    parts = ax.violinplot(data, positions=positions, widths=widths,
                          showmeans=False, showmedians=False, showextrema=False)
    if colors is None:
        colors = ['#2a4f6c', '#4682B4', '#a8c6de']
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(list(colors)[i % len(colors)])
        body.set_edgecolor('black'); body.set_alpha(1); body.set_linewidth(0.5)
        pos   = positions[i]
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.minimum(verts[:, 0], pos)
        body.get_paths()[0].vertices = verts
    return parts


def _style_robustness_ax(ax, positions, xlim_pad=0.6):
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5);  ax.spines['bottom'].set_linewidth(0.5)
    ax.set_xticks(positions); ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.tick_params(axis='x', width=0.5, length=2)
    ax.tick_params(axis='y', width=0.5, length=2)
    ax.set_xlim(positions[0] - xlim_pad, positions[-1] + xlim_pad)


def violin_fraction_x0_capped_by_ensemble(results, figsize=(1, 1), dpi=500,
                                           widths=0.7,
                                           colors=('#a8c6de', '#4682B4', '#2a4f6c'),
                                           ylim=(-0.05, 1)):
    """
    Left-half violin of IC robustness (eta_ic * S, capped at 1) per ensemble.

    Parameters
    ----------
    results : list of dicts from compute_noise_robustness.ipynb
    """
    eta_ic = np.array([r['eta_ic_crit'] for r in results], dtype=float)
    S      = np.array([r['S']           for r in results], dtype=float)
    ens    = np.array([r['ensemble_id'] for r in results])

    ensembles = np.unique(ens)[::-1]
    positions = np.arange(1, len(ensembles) + 1)
    data      = [np.minimum(eta_ic[ens == e] * S[ens == e], 1.0) for e in ensembles]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _left_half_violin(ax, data, positions, colors=colors, widths=widths)
    _style_robustness_ax(ax, positions)
    ax.set_yticks([0, 0.5, 1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.show()
    return fig, ax


def violin_eta_dyn_by_ensemble(results, figsize=(1, 1), dpi=500,
                                widths=0.7,
                                colors=('#a8c6de', '#4682B4', '#2a4f6c'),
                                ylim=(-0.05, 1)):
    """
    Left-half violin of dynamics robustness (eta_dyn_crit) per ensemble.

    Parameters
    ----------
    results : list of dicts from compute_noise_robustness.ipynb
    """
    eta_dyn = np.array([r['eta_dyn_crit'] for r in results], dtype=float)
    ens     = np.array([r['ensemble_id']  for r in results])

    ensembles = np.unique(ens)[::-1]
    positions = np.arange(1, len(ensembles) + 1)
    data      = [eta_dyn[ens == e] for e in ensembles]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _left_half_violin(ax, data, positions, colors=colors, widths=widths)
    _style_robustness_ax(ax, positions)
    ax.set_yticks([0, 0.5, 1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.show()
    return fig, ax


# =====================================================================
# Volcano plot: perturbation success heatmap
# =====================================================================
def plot_success_heatmap(p_hat, radii, cosines, figsize=(1, 1), dpi=500):
    """
    Heatmap of P(same orbit) over a grid of perturbation magnitude (r)
    and cosine angle to the reference direction (cos).

    Parameters
    ----------
    p_hat   : np.ndarray, shape (n_radii, n_cosines)
    radii   : array-like of float — perturbation magnitudes (y axis)
    cosines : array-like of float — cosine values (x axis)
    """
    from matplotlib.colors import LinearSegmentedColormap

    radii   = np.array(radii,   dtype=float)
    cosines = np.array(cosines, dtype=float)

    cmap = LinearSegmentedColormap.from_list(
        'coolwarm_trunc', plt.cm.coolwarm(np.linspace(0, 1, 256)))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(p_hat, origin='lower', aspect='auto',
                   extent=[cosines[0], cosines[-1], radii[0], radii[-1]],
                   vmin=0, vmax=1, cmap=cmap, alpha=0.3)
    ax.set_yticks([0.1, 1, 2]); ax.set_yticklabels([])
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels([])
    ax.tick_params(length=2, width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    plt.show()

    # standalone colorbar
    fig_cb, ax_cb = plt.subplots(figsize=(0.05, 1), dpi=dpi)
    cbar = fig_cb.colorbar(im, cax=ax_cb, orientation='vertical')
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.5)
    plt.show()