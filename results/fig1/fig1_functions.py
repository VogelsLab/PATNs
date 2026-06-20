import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from joblib import Parallel, delayed


# =====================================================================
# Sigmoid and its derivative
# (Note: f in utils.py is the same sigmoid but uses scipy.special.expit
#  for numerical stability. These scalar versions are kept here for
#  use in the linearization / delta decomposition below.)
# =====================================================================
def sigmoid(u, a, b, c):
    return c / (1 + np.exp(-a * (u - b))) - c / (1 + np.exp(a * b))

def sigmoid_derivative(u, a, b, c):
    exp_term = np.exp(-a * (u - b))
    return (a * c * exp_term) / (1 + exp_term)**2


# =====================================================================
# Linear / nonlinear decomposition of dx/dt along a trajectory
# =====================================================================
def compute_deltas(r_traj, W, a, b, c):
    """
    Decompose the rate of change into linear and nonlinear parts.

    Parameters
    ----------
    r_traj : np.ndarray, shape (T, N)
    W      : np.ndarray, shape (N, N)
    a, b, c : float  — sigmoid parameters (scalar, same for all neurons)

    Returns
    -------
    dl : np.ndarray, shape (N, T)  — linear term:           -r + f'(0) W r
    dn : np.ndarray, shape (N, T)  — nonlinear correction:  f(Wr) - f'(0) Wr
    """
    T, N = r_traj.shape
    fprime0 = sigmoid_derivative(0, a, b, c)
    dl = np.zeros((N, T))
    dn = np.zeros((N, T))
    for i in range(T):
        r = r_traj[i]
        u = W @ r
        dn[:, i] = sigmoid(u, a, b, c) - fprime0 * u
        dl[:, i] = -r + fprime0 * u
    return dl, dn


# =====================================================================
# Amplification landscape on a 2D grid (parallelized)
# =====================================================================
def compute_amplification_grid(J, n=500, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1), n_jobs=-1):
    """
    Evaluate proxy amplification from J_dynamics on a 2D initial-condition grid.

    Parameters
    ----------
    J      : np.ndarray, shape (2, 2)
    n      : int   — grid resolution per axis
    xlim, ylim : tuple — grid extent
    n_jobs : int   — parallel workers (-1 = all cores)

    Returns
    -------
    X, Y : np.ndarray, shape (n, n)  — meshgrid coordinates
    F    : np.ndarray, shape (n, n)  — proxy amplification at each point
    """
    from utils import J_dynamics

    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))

    def amp_wrapper(p):
        r0 = p[:, None]
        _, ampl, _, _ = J_dynamics(J, r0)
        return ampl

    F_flat = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(amp_wrapper)(p) for p in points
    )
    return X, Y, np.asarray(F_flat).reshape(n, n)


# =====================================================================
# Nonlinear vector field (for phase portrait / streamplot)
# =====================================================================
def vf_r(x, y, W, a, b, c):
    """Point-wise nonlinear vector field: dr/dt = -r + f(Wr)."""
    from utils import f
    r = np.array([x, y])
    return -r + f(W @ r, a, b, c)

def vf_grid(X, Y, W, a, b, c):
    """
    Vectorized nonlinear vector field on a meshgrid.

    Returns
    -------
    U, V : np.ndarray, shape (Ny, Nx)
    """
    from utils import f
    R = np.stack([X, Y], axis=-1)   # (Ny, Nx, 2)
    H = R @ W.T                      # (Ny, Nx, 2)
    F = -R + f(H, a, b, c)
    return F[..., 0], F[..., 1]


# =====================================================================
# Phase portrait: streamplot + trajectory overlay
# =====================================================================
def plot_phase_portrait(W, a, b, c, r_traj,
                        xlim=(-1., 1.), ylim=(-1., 1.), grid_n=401,
                        density=0.5, mask_radius=0.03,
                        seeds=None,
                        figsize=(1, 1), dpi=1000):
    """
    Streamplot of the nonlinear vector field with a trajectory overlaid.

    Parameters
    ----------
    W, a, b, c   : connectivity and sigmoid parameters
    r_traj       : np.ndarray, shape (T, 2) — trajectory to overlay
    xlim, ylim   : plot window
    grid_n       : streamplot grid resolution
    density      : streamplot density
    mask_radius  : radius around trajectory points where field is masked
    seeds        : np.ndarray, shape (K, 2) or None — extra streamline seeds
    """
    xs = np.linspace(*xlim, grid_n)
    ys = np.linspace(*ylim, grid_n)
    X, Y = np.meshgrid(xs, ys)
    U, V = vf_grid(X, Y, W, a, b, c)

    speed = np.hypot(U, V)
    eps = 1e-9
    Un = U / np.maximum(speed, eps)
    Vn = V / np.maximum(speed, eps)

    norm = plt.Normalize(vmin=np.min(speed) * 1.2, vmax=np.max(speed) * 2.5)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # --- trajectory overlay with speed-mapped color ---
    dr = np.diff(r_traj, axis=0)
    traj_speed = np.linalg.norm(dr, axis=1)
    traj_cmap = LinearSegmentedColormap.from_list('pinkpurple', ['#4B0082', '#FF69B4'])
    pts  = r_traj[:, :2]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=traj_cmap,
                        norm=plt.Normalize(traj_speed.min(), traj_speed.max()),
                        linewidth=0.6, alpha=1, zorder=5)
    lc.set_array(traj_speed)
    ax.add_collection(lc)

    # --- mask field near trajectory ---
    for pt in r_traj:
        near = np.sqrt((X - pt[0])**2 + (Y - pt[1])**2) < mask_radius
        Un[near] = np.nan
        Vn[near] = np.nan

    # --- streamplot ---
    sp = ax.streamplot(xs, ys, Un, Vn,
                       density=[density],
                       linewidth=0.2 * speed + 0.3,
                       color=speed, cmap='copper', norm=norm,
                       arrowsize=0.25, arrowstyle='-|>',
                       minlength=0.1)
    sp.lines.set_alpha(0.4)

    # --- optional seeded streamlines ---
    if seeds is not None:
        sp2 = ax.streamplot(xs, ys, U, V, start_points=seeds,
                            density=1.0,
                            linewidth=0.2 * speed + 0.3,
                            color=speed, cmap='copper', norm=norm,
                            arrowsize=0.25, arrowstyle='-|>')
        sp2.lines.set_alpha(0.4)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.3)
    plt.tight_layout()
    return fig, ax


# =====================================================================
# Plotting helpers
# =====================================================================
def style_axes(ax, border_lw=0.5):
    """Apply consistent spine and tick styling."""
    ax.tick_params(width=border_lw, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(border_lw)


def plot_standalone_colorbar(norm, cmap='RdBu_r', figsize=(0.3, 1.0),
                             dpi=500, cbar_label=' ', border_lw=0.5):
    """
    Render a standalone colorbar figure (no data axes).

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    img = ax.imshow([[0, 1]], cmap=cmap, norm=norm)
    ax.set_visible(False)
    cbar = fig.colorbar(img, ax=ax, orientation='vertical',
                        ticks=[], shrink=1, pad=0.2)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)
    cbar.outline.set_linewidth(border_lw)
    return fig, ax


def plot_heatmap(data, tt, norm, *, figsize=(1, 0.5), dpi=500, border_lw=0.5,
                 cmap='RdBu_r', interpolation='nearest',
                 separator_y=None, separator_color='black', separator_lw=0.3):
    """
    Render a single heatmap panel.

    Parameters
    ----------
    data        : np.ndarray, shape (N, T)
    tt          : np.ndarray, shape (T,)  — time axis for extent
    norm        : matplotlib Normalize instance
    separator_y : float or None — draw a horizontal line at this y position

    Returns
    -------
    fig, ax
    """
    N = data.shape[0]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.imshow(data, aspect='auto', cmap=cmap, norm=norm,
              interpolation=interpolation,
              extent=[tt[0], tt[-1], 0, N])
    ax.set_yticks([]); ax.set_xticks([])
    style_axes(ax, border_lw=border_lw)
    if separator_y is not None:
        ax.axhline(y=separator_y, color=separator_color, linewidth=separator_lw)
    return fig, ax


def make_figures_color_from_ampl(
    r_nonlinear_ampl, r_nonlinear_nonampl, W, a, b, c, t,
    i0=0, i1=700,
    figsize=(1, 0.5), dpi=500, border_lw=0.5,
    cmap='RdBu_r', interpolation='nearest',
    separator_y=None, separator_color='black', separator_lw=0.3
):
    """
    Produce 4 heatmap panels + 1 colorbar for the linear/nonlinear decomposition:
      - dl_ampl, dn_ampl  (amplifying initial condition)
      - dl_non,  dn_non   (non-amplifying initial condition)

    Color scale is determined from the amplifying panels and reused for all four.

    Parameters
    ----------
    r_nonlinear_ampl, r_nonlinear_nonampl : np.ndarray, shape (T, N)
    W                                     : np.ndarray, shape (N, N)
    a, b, c                               : float — sigmoid parameters
    t                                     : np.ndarray — time vector
    i0, i1                                : int — time slice indices
    """
    N = r_nonlinear_ampl.shape[1]

    dl_ampl, dn_ampl = compute_deltas(r_nonlinear_ampl, W, a, b, c)
    dl_non,  dn_non  = compute_deltas(r_nonlinear_nonampl, W, a, b, c)

    dl_ampl_s = dl_ampl[:, i0:i1]
    dn_ampl_s = dn_ampl[:, i0:i1]
    dl_non_s  = dl_non[:,  i0:i1]
    dn_non_s  = dn_non[:,  i0:i1]
    tt = t[i0:i1]

    # color scale from amplifying panels only
    vlim = float(np.max(np.abs(np.concatenate([dl_ampl_s.ravel(), dn_ampl_s.ravel()]))))
    if vlim == 0:
        vlim = 1e-9
    shared_norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    plot_standalone_colorbar(shared_norm, cmap=cmap, dpi=dpi, border_lw=border_lw)
    plot_heatmap(dl_ampl_s, tt, shared_norm, figsize=figsize, dpi=dpi,
                 border_lw=border_lw, cmap=cmap, interpolation=interpolation,
                 separator_y=separator_y, separator_color=separator_color,
                 separator_lw=separator_lw)
    plot_heatmap(dn_ampl_s, tt, shared_norm, figsize=figsize, dpi=dpi,
                 border_lw=border_lw, cmap=cmap, interpolation=interpolation,
                 separator_y=separator_y, separator_color=separator_color,
                 separator_lw=separator_lw)
    plot_heatmap(dl_non_s, tt, shared_norm, figsize=figsize, dpi=dpi,
                 border_lw=border_lw, cmap=cmap, interpolation=interpolation,
                 separator_y=separator_y, separator_color=separator_color,
                 separator_lw=separator_lw)
    plot_heatmap(dn_non_s, tt, shared_norm, figsize=figsize, dpi=dpi,
                 border_lw=border_lw, cmap=cmap, interpolation=interpolation,
                 separator_y=separator_y, separator_color=separator_color,
                 separator_lw=separator_lw)
    plt.show()
